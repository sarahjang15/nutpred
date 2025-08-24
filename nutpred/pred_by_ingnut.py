import re
import numpy as np
import pandas as pd
import cvxpy as cp
from numpy.linalg import norm
from typing import List, Dict, Tuple, Optional
import logging
import traceback
from .metrics import r2_manual, smape

logger = logging.getLogger(__name__)

# Threshold to separate salted vs unsalted butter using snack Na/kcal
BUTTER_NA_PER_KCAL_THRESHOLD = 0.4  # mg/kcal

# ---------------------------------------------------------------------
# Helpers for grouping and variant selection
# ---------------------------------------------------------------------
def _build_groups(ingnut_df: pd.DataFrame, base_col_candidates: List[str]):
    """
    Return:
      - groups: dict base_norm -> row indices (variants),
      - base_col: chosen base column name,
      - case_col: name if present else None
    """
    logger.debug("Building ingredient groups from base column candidates")
    base_col = next((c for c in base_col_candidates if c in ingnut_df.columns), None)
    if base_col is None:
        logger.error(f"ingnut_df missing any of {base_col_candidates}")
        raise ValueError(f"ingnut_df missing any of {base_col_candidates}")
    
    logger.info(f"Using base column: {base_col}")
    base_norm = ingnut_df[base_col].astype(str).str.upper().str.strip()
    ingnut_df["_base_norm"] = base_norm
    case_col = "case" if "case" in ingnut_df.columns else None
    groups = ingnut_df.groupby("_base_norm").indices
    
    logger.info(f"Built {len(groups)} ingredient groups")
    if case_col:
        logger.info(f"Case column found: {case_col}")
    else:
        logger.info("No case column found")
    
    return groups, base_col, case_col

def _find_butter_variant_by_saltness(idxs, want_salted, ingnut_df, case_col):
    """Choose salted vs unsalted butter based on snack Na/kcal ratio."""
    try:
        logger.debug(f"Resolving butter variant for indices {idxs}, want_salted={want_salted}")
        
        # Convert integer indices to actual DataFrame indices
        if isinstance(idxs, np.ndarray) and idxs.dtype == np.int64:
            # These are integer positions, convert to actual indices
            actual_indices = ingnut_df.index[idxs]
        else:
            actual_indices = idxs
        
        # Check if indices exist in the dataframe
        if not all(idx in ingnut_df.index for idx in actual_indices):
            logger.warning(f"Some indices not found in ingnut_df index")
            # Return first available index
            available_idxs = [idx for idx in actual_indices if idx in ingnut_df.index]
            if available_idxs:
                return available_idxs[0]
            else:
                return actual_indices[0] if len(actual_indices) > 0 else None
        
        # Get cases for the available indices
        cases = ingnut_df.loc[actual_indices, case_col].astype(str).str.lower().fillna("")
        logger.debug(f"Available cases: {cases.tolist()}")
        
        # Find salted/unsalted variants
        salted_mask = cases.str.contains("salted|salt")
        unsalted_mask = cases.str.contains("unsalted|no salt|without salt")
        
        if want_salted:
            salted_idxs = actual_indices[salted_mask]
            if len(salted_idxs) > 0:
                logger.debug(f"Selected salted butter: {salted_idxs[0]}")
                return salted_idxs[0]
        else:
            unsalted_idxs = actual_indices[unsalted_mask]
            if len(unsalted_idxs) > 0:
                logger.debug(f"Selected unsalted butter: {unsalted_idxs[0]}")
                return unsalted_idxs[0]
        
        # Fallback to first variant
        logger.debug(f"Using fallback variant: {actual_indices[0]}")
        return actual_indices[0]
        
    except Exception as e:
        logger.error(f"Error in _find_butter_variant_by_saltness: {e}")
        return actual_indices[0] if len(actual_indices) > 0 else None

# ---------------------------------------------------------------------
# Core predictor
# ---------------------------------------------------------------------
def predict_ingnut_weights_and_targets(
    snack: pd.DataFrame,
    ingnut_df: pd.DataFrame,
    nut8_cols: List[str],
    ingnut_cols: List[str],
    resolver: str = "rule",           # Only "rule" supported now
    constraint: str = "nnls_mono",
    ridge: float = 0.0,
    robust: bool = False,
    solver_name: str = "osqp",
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Returns (snack_with_pred_cols, weights_matrix, variant_universe)
    Adds: pred_{nut8}, pred_Calcium(mg), pred_Fiber(g), pred_Iron(mg) if available in ingnut_df.
    """
    try:
        logger.info(f"Starting ingredient-nutrient prediction with {len(snack)} samples, {len(ingnut_cols)} features")
        logger.info(f"Parameters: resolver={resolver}, constraint={constraint}, ridge={ridge}, robust={robust}, solver={solver_name}")
        
        # Create column mapping between snack and ingnut columns
        column_mapping = {}
        for nut_col, ingnut_col in zip(nut8_cols, ingnut_cols):
            if nut_col in snack.columns and ingnut_col in ingnut_df.columns:
                column_mapping[nut_col] = ingnut_col
        
        if len(column_mapping) == 0:
            logger.error("No matching nutrients found between snack and ingnut data")
            logger.error(f"Snack columns: {[c for c in nut8_cols if c in snack.columns]}")
            logger.error(f"Ingnut columns: {[c for c in ingnut_cols if c in ingnut_df.columns]}")
            raise ValueError("No matching nutrients found")
        
        # Use only available nutrients
        nut8_cols = list(column_mapping.keys())
        ingnut_cols = list(column_mapping.values())
        
        logger.info(f"Column mapping: {column_mapping}")
        logger.info(f"Final nutrient columns: {nut8_cols}")
        
        # Validate column count match
        if len(nut8_cols) != len(ingnut_cols):
            logger.error(f"Column count mismatch: snack={len(nut8_cols)}, ingnut={len(ingnut_cols)}")
            logger.error(f"Snack columns: {nut8_cols}")
            logger.error(f"Ingnut columns: {ingnut_cols}")
            raise ValueError("Column count mismatch")
        
        # Create variant_key index for ingnut_df
        logger.info("Creating variant_key index for ingnut_df")
        if "variant_key" not in ingnut_df.columns:
            # Try to create from base column
            base_cols = [col for col in ingnut_df.columns if "ing" in col.lower() or "name" in col.lower()]
            if base_cols:
                base_col = base_cols[0]
                logger.info(f"Created variant_key from base column: {base_col}")
                ingnut_df = ingnut_df.copy()
                ingnut_df["variant_key"] = ingnut_df[base_col].astype(str)
            else:
                logger.warning("No suitable base column found, using index")
                ingnut_df = ingnut_df.copy()
                ingnut_df["variant_key"] = ingnut_df.index.astype(str)
        
        # Set variant_key as index
        ingnut_df = ingnut_df.set_index("variant_key")
        logger.info(f"Set variant_key as index. Total variants: {len(ingnut_df)}")
        
        # Prepare data matrices using column mapping
        X_all = ingnut_df[ingnut_cols].values  # (K, D) nutrient matrix from ingnut data
        Y_mat = snack[nut8_cols].values        # (N, D) target matrix from snack data
        
        logger.info(f"Data shapes: X_all={X_all.shape}, Y_mat={Y_mat.shape}, K={len(ingnut_df)} variants")
        
        # Build ingredient groups
        groups, base_col, case_col = _build_groups(ingnut_df, [col for col in ingnut_df.columns if "ing" in col.lower()])
        logger.info(f"Built {len(groups)} ingredient groups")
        
        # Check for case column for butter rule
        if case_col:
            logger.info(f"Case column found: {case_col}")
            # Check if we have the required nutrients for butter rule
            if "Sodium(mg)" in nut8_cols and "Energy(kcal)" in nut8_cols:
                logger.info("Found Sodium(mg) in nut8_cols for butter rule")
                logger.info("Found Energy(kcal) in nut8_cols for butter rule")
            else:
                logger.warning("Missing required nutrients for butter rule, will use fallback")
        
        # Create name to index mapping
        name_to_idx = {name: idx for idx, name in enumerate(ingnut_df.index)}
        logger.info(f"Created name_to_idx mapping for {len(name_to_idx)} variants")
        
        def _resolve_variant(token: str, j: int):
            """Resolve a token to a specific variant index."""
            try:
                # Direct match to a variant_key
                if token in name_to_idx:
                    return name_to_idx[token]
                
                base = str(token).upper().strip()
                
                if resolver == "rule" and base in groups:
                    # Special rule: BUTTER salted vs unsalted
                    if base == "BUTTER" and "Sodium(mg)" in nut8_cols and "Energy(kcal)" in nut8_cols:
                        sodium_idx = nut8_cols.index("Sodium(mg)")
                        energy_idx = nut8_cols.index("Energy(kcal)")
                        na = float(Y_mat[j, sodium_idx])
                        kcal = float(Y_mat[j, energy_idx])
                        if np.isfinite(na) and np.isfinite(kcal) and kcal > 0:
                            na_per_kcal = na / kcal
                            want_salted = na_per_kcal >= BUTTER_NA_PER_KCAL_THRESHOLD
                            logger.debug(f"Sample {j} butter: Na={na}, kcal={kcal}, Na/kcal={na_per_kcal:.4f}, want_salted={want_salted}")
                            
                            pick = _find_butter_variant_by_saltness(groups[base], want_salted, ingnut_df, case_col)
                            if pick is not None:
                                return name_to_idx.get(pick, None)
                    
                    # Default rule fallback: first variant in group
                    logger.debug(f"Using first variant for {base}")
                    return groups[base][0]
                
                # No rule match found
                logger.debug(f"No rule match found for token '{token}'")
                return None
                
            except Exception as e:
                logger.error(f"Error in _resolve_variant for token '{token}': {e}")
                return None
        
        # Map ingredients to variants
        logger.info("Mapping ingredients to variants...")
        mapped_tokens = []
        progress_step = max(1, len(snack) // 100)  # Progress every 1%
        
        for j, (_, row) in enumerate(snack.iterrows()):
            if j % progress_step == 0:
                logger.info(f"Mapping progress: {j}/{len(snack)} samples processed, {len(mapped_tokens)} tokens mapped")
            
            tokens = row.get("mapped_list", [])
            if not isinstance(tokens, list):
                tokens = []
            
            row_tokens = []
            for token in tokens:
                if token in name_to_idx:
                    idx = _resolve_variant(token, j)
                    if idx is not None:
                        row_tokens.append(idx)
            
            mapped_tokens.append(row_tokens)
        
        logger.info(f"Mapping complete: {len(mapped_tokens)} samples processed")
        
        # Choose solver
        solver_map = {"osqp": cp.OSQP, "clarabel": cp.CLARABEL, "scs": cp.SCS, "ecos": cp.ECOS, "piqp": cp.PIQP}
        if solver_name not in solver_map:
            logger.warning(f"Solver {solver_name} not available, falling back to OSQP")
            solver_name = "osqp"
        solver_enum = solver_map[solver_name]
        solver_kwargs = dict(eps_abs=1e-6, eps_rel=1e-6, verbose=False, max_iter=100000, polish=True)
        logger.info(f"Using solver: {solver_name}")
        
        # Prepare optimization matrices
        K = len(ingnut_df)  # Number of variants
        n = len(snack)      # Number of samples
        X_full = X_all.T    # (D, K) - transpose for optimization
        
        preds_w = np.zeros((n, K))
        failed = []
        
        def _loss(expr, v):
            base = cp.sum_squares(expr) if not robust else cp.sum(cp.huber(expr, 1.0))
            return base + (ridge * cp.sum_squares(v) if ridge > 0 else 0)
        
        # Solve per snack
        logger.info("Starting optimization for each sample...")
        for j, Sj in enumerate(mapped_tokens):
            if j % 50 == 0:  # Progress update every 50 samples
                logger.info(f"Optimization progress: {j}/{len(mapped_tokens)} samples completed, {len(failed)} failed")
                
            if not Sj:
                logger.debug(f"Sample {j}: No ingredients to optimize")
                continue
                
            Kj = len(Sj)
            Xj = X_full[:, Sj]
            v = cp.Variable(Kj, nonneg=True)
            
            cons = []
            if constraint in ("eq1", "eq1_mono"):
                cons.append(cp.sum(v) == 1)
            elif constraint in ("le1", "le1_mono"):
                cons.append(cp.sum(v) <= 1)
            if constraint.endswith("_mono") and Kj > 1:
                cons += [v[k] >= v[k+1] for k in range(Kj-1)]
            
            yj = Y_mat[j]
            prob = cp.Problem(cp.Minimize(_loss(Xj @ v - yj, v)), cons)
            
            try:
                prob.solve(solver=solver_enum, **solver_kwargs)
                if prob.status not in ["optimal", "optimal_inaccurate"]:
                    logger.warning(f"Sample {j}: Solver status '{prob.status}', using fallback")
                    raise RuntimeError(prob.status)
                wj = v.value
                logger.debug(f"Sample {j}: Optimization successful with {Kj} ingredients")
            except Exception as e:
                logger.warning(f"Sample {j}: Optimization failed ({str(e)}), using uniform weights")
                failed.append(j)
                wj = np.ones(Kj) / Kj
            
            full = np.zeros(K)
            full[Sj] = wj
            preds_w[j] = full
        
        logger.info(f"Optimization complete. {len(failed)} samples failed, using fallback weights")
        
        # Predict base nutrients
        Y_pred = preds_w @ X_full.T
        for j, c in enumerate(nut8_cols):
            snack[f"pred_{c}"] = Y_pred[:, j]
        logger.info(f"Predicted base nutrients: {nut8_cols}")
        
        # Predict targets with same weights if available
        extra_map = {'Calcium, Ca': 'pred_Calcium(mg)', 'Fiber, total dietary': 'pred_Fiber(g)', 'Iron, Fe': 'pred_Iron(mg)'}
        predicted_targets = []
        for src, dst in extra_map.items():
            if src in ingnut_df.columns:
                X_extra = ingnut_df[src].values.reshape(1, -1)
                snack[dst] = (preds_w @ X_extra.T).ravel()
                predicted_targets.append(dst)
        
        if predicted_targets:
            logger.info(f"Predicted target nutrients: {predicted_targets}")
        else:
            logger.warning("No target nutrients found in ingnut_df")
        
        logger.info("Ingredient-nutrient prediction complete")
        return snack, preds_w, list(ingnut_df.index)
        
    except Exception as e:
        logger.error(f"Error in predict_ingnut_weights_and_targets: {e}")
        logger.error(traceback.format_exc())
        raise

# ---------------------------------------------------------------------
# Metrics for ing_pred
# ---------------------------------------------------------------------
def metrics_ing_pred(snack: pd.DataFrame) -> pd.DataFrame:
    """Metrics for ing_pred method (direct predictions from optimizer)."""
    logger.info("Calculating metrics for ingredient prediction method")
    
    # Create first_mapped filter
    def is_first_mapped(mapped_list):
        if not isinstance(mapped_list, list) or len(mapped_list) == 0:
            return False
        # Check if the first ingredient is mapped (not empty string)
        return mapped_list[0].strip() != ""
    
    first_mapped_mask = snack['mapped_list'].apply(is_first_mapped)
    logger.info(f"First mapped samples: {first_mapped_mask.sum()}/{len(snack)}")
    
    rows = []
    pairs = {
        "Calcium(mg)": "pred_Calcium(mg)",
        "Fiber(g)": "pred_Fiber(g)",
        "Iron(mg)": "pred_Iron(mg)",
    }
    
    for truth, predc in pairs.items():
        if truth in snack.columns and predc in snack.columns:
            y = snack[truth].to_numpy(float)
            yhat = snack[predc].to_numpy(float)
            
            # Filter for first_mapped and non-NaN values
            valid_mask = first_mapped_mask & ~(np.isnan(y) | np.isnan(yhat))
            if not np.any(valid_mask):
                logger.warning(f"No valid data for {truth} (first_mapped)")
                continue
                
            y_valid = y[valid_mask]
            yhat_valid = yhat[valid_mask]
            
            r2 = r2_manual(y_valid, yhat_valid)
            rmse = float(np.sqrt(np.mean((y_valid - yhat_valid)**2)))
            mae = float(np.mean(np.abs(y_valid - yhat_valid)))
            mape = float(np.mean(np.abs((y_valid - yhat_valid) / (np.abs(y_valid) + 1e-6))))
            s = smape(y_valid, yhat_valid)
            
            logger.info(f"{truth} (first_mapped): R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, SMAPE={s:.4f}%")
            
            rows.append({"Nutrient": truth, "FeatureSet": "ing_pred", "Model": "Optimization",
                         "R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape, "SMAPE": s})
        else:
            logger.warning(f"Missing columns for {truth}: truth={truth in snack.columns}, pred={predc in snack.columns}")
    
    result_df = pd.DataFrame(rows)
    logger.info(f"Metrics calculation complete. {len(result_df)} nutrients evaluated")
    return result_df
