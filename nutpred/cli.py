#!/usr/bin/env python3
"""
nutpred CLI — supports:
  - python -m nutpred.cli
  - python nutpred/cli.py
"""

import os
import argparse
import numpy as np
import pandas as pd
import logging
import traceback
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- allow running as a script: python nutpred/cli.py --------------------------
if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- absolute imports from the package ----------------------------------------
from nutpred.preprocess import (
    load_inputs,
    ensure_mapped_list_column,
    filter_rows,
    make_topk,
    build_binary_and_scores,
    ensure_umap_columns,
    select_base_nutrients,
    ensure_targets,
)
from nutpred.pred_by_ingnut import (
    predict_ingnut_weights_and_targets,
    metrics_ing_pred,
)
from nutpred.pred_by_fullnut import train_eval_sets
from nutpred.viz import compare_feature_sets


def parse_args():
    ap = argparse.ArgumentParser(
        description="Predict Calcium/Fiber/Iron from ingredients + nutrients."
    )
    ap.add_argument("--snack-csv", type=str, required=True)
    ap.add_argument("--thesaurus-xlsx", type=str, required=True)
    ap.add_argument("--ingnut-csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="./nutpred_outputs")
    ap.add_argument("--exclude", type=str, default="popcorn,pretzel,pretzels")
    ap.add_argument("--top-n", type=int, default=135)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--force-rf", action="store_true")

    # Optimization (ing_pred)
    ap.add_argument("--opt-resolver", type=str, default="cosine", choices=["cosine"])
    ap.add_argument(
        "--opt-constraint",
        type=str,
        default="nnls_mono",
        choices=["nnls_only", "nnls_mono", "eq1", "eq1_mono", "le1", "le1_mono"],
    )
    ap.add_argument(
        "--opt-solver",
        type=str,
        default="osqp",
        choices=["osqp", "clarabel", "scs", "ecos", "piqp"],
    )
    ap.add_argument("--opt-ridge", type=float, default=0.0)
    ap.add_argument("--opt-robust", action="store_true")
    return ap.parse_args()


def _reorder_truth_pred_columns(df: pd.DataFrame, nut8_cols, extra_map):
    """Place truth/pred pairs side by side: Sodium, pred_Sodium, Calcium, pred_Calcium, ..."""
    logger.debug("Reordering columns to place truth/pred pairs side by side")
    
    # ensure pred for base nutrients
    for c in nut8_cols:
        pc = f"pred_{c}"
        if pc not in df.columns:
            df[pc] = np.nan
            logger.debug(f"Added missing prediction column: {pc}")
    
    # ensure pred for targets
    for truth, pc in extra_map.items():
        if pc not in df.columns:
            df[pc] = np.nan
            logger.debug(f"Added missing target prediction column: {pc}")

    pairs = []
    for c in nut8_cols:
        pairs += [c, f"pred_{c}"]
    for truth, pc in extra_map.items():
        if truth in df.columns:
            pairs += [truth, pc]
        else:
            pairs += [pc]

    pair_set = set(pairs)
    other = [c for c in df.columns if c not in pair_set]
    result = df[other + pairs]
    
    logger.debug(f"Column reordering complete. Final shape: {result.shape}")
    return result


def main():
    try:
        logger.info("Starting nutpred CLI execution")
        args = parse_args()
        
        logger.info(f"Arguments: snack_csv={args.snack_csv}, thesaurus_xlsx={args.thesaurus_xlsx}, "
                    f"ingnut_csv={args.ingnut_csv}, outdir={args.outdir}")
        logger.info(f"Parameters: top_n={args.top_n}, test_size={args.test_size}, cv={args.cv}, "
                    f"opt_constraint={args.opt_constraint}, opt_solver={args.opt_solver}")
        
        os.makedirs(args.outdir, exist_ok=True)
        logger.info(f"Created output directory: {args.outdir}")

        # 1) load + preprocess
        logger.info("Step 1: Loading and preprocessing data")
        snack, thesaurus = load_inputs(args.snack_csv, args.thesaurus_xlsx)
        logger.info(f"Loaded snack data: {snack.shape}, thesaurus: {thesaurus.shape}")
        
        snack = ensure_mapped_list_column(snack, thesaurus)
        logger.info("Ensured mapped_list column")
        
        exclude_list = tuple(t.strip().lower() for t in args.exclude.split(",") if t.strip())
        logger.info(f"Filtering out: {exclude_list}")
        snack = filter_rows(snack, exclude_list)
        logger.info(f"After filtering: {snack.shape}")
        
        snack = ensure_umap_columns(snack, expected_dim=10)
        logger.info("Ensured UMAP columns")

        # 2) features (binary/score) with top-K
        logger.info("Step 2: Building feature sets")
        top_list = make_topk(snack, args.top_n)
        logger.info(f"Created top-{args.top_n} ingredient list")
        
        binary_df, score_df = build_binary_and_scores(snack, top_list, max_score=20)
        logger.info(f"Built binary features: {binary_df.shape}, score features: {score_df.shape}")
        
        snack = snack.join(binary_df).join(score_df)
        logger.info(f"Joined feature sets. Final shape: {snack.shape}")

        # 3) columns
        logger.info("Step 3: Selecting columns")
        nut8_cols = select_base_nutrients(snack)
        targets = ensure_targets(snack)
        umap_cols = [c for c in snack.columns if c.startswith("umap_10_")]
        binary_cols = [c for c in snack.columns if c.startswith("binary_")]
        score_cols = [c for c in snack.columns if c.startswith("score_")]
        
        logger.info(f"Selected columns: nut8={len(nut8_cols)}, targets={len(targets)}, "
                    f"umap={len(umap_cols)}, binary={len(binary_cols)}, score={len(score_cols)}")

        # 4) ingnut optimization → ing_pred
        logger.info("Step 4: Ingredient-nutrient optimization")
        ingnut_df = pd.read_csv(args.ingnut_csv)
        logger.info(f"Loaded ingnut data: {ingnut_df.shape}")
        
        ingnut_cols = [
            "Energy",
            "Protein",
            "Total lipid (fat)",
            "Carbohydrate, by difference",
            "Fiber, total dietary",
            "Sugars, total including NLEA",
            "Calcium, Ca",
            "Iron, Fe",
            "Sodium, Na",
            "Potassium, K",
        ]
        ingnut_cols = [c for c in ingnut_cols if c in ingnut_df.columns]
        logger.info(f"Using ingnut columns: {ingnut_cols}")
        
        logger.info("Starting ingredient-nutrient prediction...")
        snack, preds_w, variant_universe = predict_ingnut_weights_and_targets(
            snack, ingnut_df, nut8_cols, ingnut_cols,
            resolver=args.opt_resolver,
            constraint=args.opt_constraint,
            ridge=args.opt_ridge,
            robust=args.opt_robust,
            solver_name=args.opt_solver,
        )
        logger.info(f"Optimization complete. Weight matrix shape: {preds_w.shape}")

        # 5) fullnut ML → full_pred
        logger.info("Step 5: Full nutrient machine learning")
        feature_sets = {
            "umap": umap_cols,
            "binary": binary_cols,
            "score": score_cols,
            "umap_binary": umap_cols + binary_cols,
            "umap_score": umap_cols + score_cols,
            "binary_score": binary_cols + score_cols,
            "umap_binary_score": umap_cols + binary_cols + score_cols,
        }
        
        logger.info("Starting ML training...")
        full_results = train_eval_sets(
            snack, targets, feature_sets, test_size=args.test_size, cv=args.cv, force_rf=args.force_rf
        )
        logger.info(f"Full nutrient ML complete. Results for {len(full_results)} feature sets")

        # 6) metrics
        logger.info("Step 6: Calculating metrics")
        ing_metrics = metrics_ing_pred(snack)
        logger.info(f"Ingredient prediction metrics: {len(ing_metrics)} nutrients")
        
        all_metrics = pd.concat([ing_metrics, full_results], ignore_index=True)
        logger.info(f"Combined metrics: {len(all_metrics)} total rows")

        # 7) save outputs
        logger.info("Step 7: Saving outputs")
        
        # Save full snack dataframe with all predictions
        snack_ordered = _reorder_truth_pred_columns(
            snack, nut8_cols, 
            {"Calcium(mg)": "pred_Calcium(mg)", "Fiber(g)": "pred_Fiber(g)", "Iron(mg)": "pred_Iron(mg)"}
        )
        snack_ordered.to_csv(os.path.join(args.outdir, "snack_df_complete.csv"), index=False)
        logger.info("Saved snack_df_complete.csv with all predictions")
        
        # Save metrics
        all_metrics.to_csv(os.path.join(args.outdir, "metrics.csv"), index=False)
        logger.info("Saved metrics.csv")
        
        # Save weights matrix
        weights_df = pd.DataFrame(preds_w, columns=variant_universe)
        weights_df.to_csv(os.path.join(args.outdir, "ingredient_weights.csv"), index=False)
        logger.info("Saved ingredient_weights.csv")
        
        # Save ingredient lists
        ingredient_summary = pd.DataFrame({
            'ingredient_list_raw': snack['ingredient_list_raw'].apply(lambda x: str(x)),
            'mapped_list': snack['mapped_list'].apply(lambda x: str(x)),
            'top_ingredients': [', '.join(lst[:10]) if isinstance(lst, list) else '' for lst in snack['mapped_list']]
        })
        ingredient_summary.to_csv(os.path.join(args.outdir, "ingredient_parsing_summary.csv"), index=False)
        logger.info("Saved ingredient_parsing_summary.csv")
        
        # 8) visualization
        logger.info("Step 8: Creating visualizations")
        try:
            fig = compare_feature_sets(all_metrics)
            fig.savefig(os.path.join(args.outdir, "feature_comparison.png"), dpi=300, bbox_inches="tight")
            logger.info("Saved feature_comparison.png")
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            logger.error(traceback.format_exc())
        
        # 9) Summary report
        logger.info("Step 9: Creating summary report")
        summary_report = f"""
NUTPRED EXECUTION SUMMARY
========================

Input Data:
- Snack data: {snack.shape[0]} samples, {snack.shape[1]} features
- Thesaurus: {thesaurus.shape[0]} mappings
- Ingredient-nutrient data: {ingnut_df.shape[0]} variants

Processing:
- Top {args.top_n} ingredients selected
- {len(binary_cols)} binary features created
- {len(score_cols)} score features created
- {len(umap_cols)} UMAP features used

Predictions:
- Base nutrients predicted: {len(nut8_cols)}
- Target nutrients predicted: {len(targets)}
- Optimization solver: {args.opt_solver}
- ML models trained: {len(feature_sets)} feature sets

Output Files:
- snack_df_complete.csv: Full dataset with predictions
- metrics.csv: Performance metrics for all models
- ingredient_weights.csv: Optimization weights
- ingredient_parsing_summary.csv: Ingredient parsing results
- feature_comparison.png: Visualization

Execution completed successfully!
        """
        
        with open(os.path.join(args.outdir, "execution_summary.txt"), "w") as f:
            f.write(summary_report)
        logger.info("Saved execution_summary.txt")
        
        print(summary_report)
        logger.info("nutpred CLI execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error(traceback.format_exc())
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
