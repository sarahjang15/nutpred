#!/usr/bin/env python3
"""
Test script for nutpred to identify where the process might be hanging
"""

import os
import sys
import logging
import traceback
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_step_by_step():
    """Test each step of the nutpred pipeline separately"""
    
    try:
        logger.info("=== STEP 1: Import modules ===")
        from nutpred.preprocess import (
            load_inputs, ensure_mapped_list_column, filter_rows, make_topk,
            build_binary_and_scores, ensure_umap_columns, select_base_nutrients, ensure_targets
        )
        logger.info("All preprocessing modules imported successfully")
        
        from nutpred.pred_by_ingnut import predict_ingnut_weights_and_targets, metrics_ing_pred
        logger.info("Ingredient prediction module imported successfully")
        
        from nutpred.pred_by_fullnut import train_eval_sets
        logger.info("Full nutrient prediction module imported successfully")
        
        logger.info("=== STEP 2: Load data ===")
        snack, thesaurus = load_inputs("data/snack_input_df.csv", "data/THESAURUSFORPUBLICRELEASE.XLSX")
        logger.info(f"Loaded snack data: {snack.shape}")
        logger.info(f"Loaded thesaurus: {thesaurus.shape}")
        
        logger.info("=== STEP 3: Preprocess data ===")
        snack = ensure_mapped_list_column(snack, thesaurus)
        logger.info("Created mapped_list column")
        
        snack = filter_rows(snack, ("popcorn", "pretzel", "pretzels"))
        logger.info(f"Filtered data: {snack.shape}")
        
        snack = ensure_umap_columns(snack, expected_dim=10)
        logger.info("Ensured UMAP columns")
        
        logger.info("=== STEP 4: Build features ===")
        top_list = make_topk(snack, 135)
        logger.info(f"Created top-135 ingredient list")
        
        binary_df, score_df = build_binary_and_scores(snack, top_list, max_score=20)
        logger.info(f"Built binary features: {binary_df.shape}")
        logger.info(f"Built score features: {score_df.shape}")
        
        snack = snack.join(binary_df).join(score_df)
        logger.info(f"Joined feature sets: {snack.shape}")
        
        logger.info("=== STEP 5: Select columns ===")
        nut8_cols = select_base_nutrients(snack)
        targets = ensure_targets(snack)
        umap_cols = [c for c in snack.columns if c.startswith("umap_10_")]
        binary_cols = [c for c in snack.columns if c.startswith("binary_")]
        score_cols = [c for c in snack.columns if c.startswith("score_")]
        
        logger.info(f"Selected columns: nut8={len(nut8_cols)}, targets={len(targets)}")
        logger.info(f"Feature columns: umap={len(umap_cols)}, binary={len(binary_cols)}, score={len(score_cols)}")
        
        logger.info("=== STEP 6: Load ingnut data ===")
        ingnut_df = pd.read_csv("data/ingnut_df_top135.csv")
        logger.info(f"Loaded ingnut data: {ingnut_df.shape}")
        
        ingnut_cols = [
            "Energy", "Protein", "Total lipid (fat)", "Carbohydrate, by difference",
            "Fiber, total dietary", "Sugars, total including NLEA", "Calcium, Ca",
            "Iron, Fe", "Sodium, Na", "Potassium, K",
        ]
        ingnut_cols = [c for c in ingnut_cols if c in ingnut_df.columns]
        logger.info(f"Using ingnut columns: {ingnut_cols}")
        
        logger.info("=== STEP 7: Test ingredient prediction (small subset) ===")
        # Test with a small subset first
        test_snack = snack.head(10).copy()
        logger.info(f"Testing with {len(test_snack)} samples")
        
        try:
            test_snack, preds_w, variant_universe = predict_ingnut_weights_and_targets(
                test_snack, ingnut_df, nut8_cols, ingnut_cols,
                resolver="rule", constraint="nnls_mono", ridge=0.0, 
                robust=False, solver_name="osqp"
            )
            logger.info("Ingredient prediction successful on test subset")
            logger.info(f"Weight matrix shape: {preds_w.shape}")
        except Exception as e:
            logger.error(f"Ingredient prediction failed on test subset: {e}")
            logger.error(traceback.format_exc())
            return False
        
        logger.info("=== STEP 8: Full ingredient prediction ===")
        try:
            snack, preds_w, variant_universe = predict_ingnut_weights_and_targets(
                snack, ingnut_df, nut8_cols, ingnut_cols,
                resolver="rule", constraint="nnls_mono", ridge=0.0, 
                robust=False, solver_name="osqp"
            )
            logger.info("Full ingredient prediction successful")
        except Exception as e:
            logger.error(f"Full ingredient prediction failed: {e}")
            logger.error(traceback.format_exc())
            return False
        
        logger.info("=== STEP 9: Test ML training (small subset) ===")
        feature_sets = {
            "umap": umap_cols,
            "binary": binary_cols,
            "score": score_cols,
        }
        
        try:
            full_results = train_eval_sets(
                test_snack, targets, feature_sets, test_size=0.5, cv=2, force_rf=False
            )
            logger.info("ML training successful")
            logger.info(f"Generated {len(full_results)} results")
        except Exception as e:
            logger.error(f"ML training failed: {e}")
            logger.error(traceback.format_exc())
            return False
        
        logger.info("=== STEP 10: Test metrics calculation ===")
        try:
            ing_metrics = metrics_ing_pred(test_snack)
            logger.info(f"Metrics calculation successful: {len(ing_metrics)} nutrients")
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            logger.error(traceback.format_exc())
            return False
        
        logger.info("=== STEP 11: Test output saving ===")
        try:
            os.makedirs("test_outputs", exist_ok=True)
            test_snack.to_csv("test_outputs/test_snack.csv", index=False)
            pd.DataFrame(preds_w, columns=variant_universe).to_csv("test_outputs/test_weights.csv", index=False)
            ing_metrics.to_csv("test_outputs/test_metrics.csv", index=False)
            logger.info("All outputs saved successfully")
        except Exception as e:
            logger.error(f"Output saving failed: {e}")
            logger.error(traceback.format_exc())
            return False
        
        logger.info("=== ALL TESTS PASSED ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_step_by_step()
    if success:
        print("\nAll tests passed successfully!")
    else:
        print("\nTests failed. Check the logs above for details.")
