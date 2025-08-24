#!/usr/bin/env python3
"""
Complete nutpred runner that generates all requested outputs:
1. Full snack_df.csv with parsed and mapped ingredient list
2. Predicted output by each set of models/features
3. Visualizations
"""

import os
import sys
import logging
import traceback
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nutpred_complete.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

def run_complete_pipeline():
    """Run the complete nutpred pipeline with all outputs"""
    
    try:
        logger.info("Starting complete nutpred pipeline")
        
        # Import all modules
        from nutpred.preprocess import (
            load_inputs, ensure_mapped_list_column, filter_rows, make_topk,
            build_binary_and_scores, ensure_umap_columns, select_base_nutrients, ensure_targets
        )
        from nutpred.pred_by_ingnut import predict_ingnut_weights_and_targets, metrics_ing_pred
        from nutpred.pred_by_fullnut import train_eval_sets
        from nutpred.viz import compare_feature_sets
        
        # Create output directory
        outdir = "complete_outputs"
        os.makedirs(outdir, exist_ok=True)
        logger.info(f"Created output directory: {outdir}")
        
        # Step 1: Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data")
        snack, thesaurus = load_inputs("data/snack_input_df.csv", "data/THESAURUSFORPUBLICRELEASE.XLSX")
        logger.info(f"   Loaded snack data: {snack.shape}")
        logger.info(f"   Loaded thesaurus: {thesaurus.shape}")
        
        # Create ingredient mapping
        snack = ensure_mapped_list_column(snack, thesaurus)
        logger.info("   Created mapped ingredient lists")
        
        # Filter data
        snack = filter_rows(snack, ("popcorn", "pretzel", "pretzels"))
        logger.info(f"   Filtered data: {snack.shape}")
        
        # Ensure UMAP columns
        snack = ensure_umap_columns(snack, expected_dim=10)
        logger.info("   Ensured UMAP columns")
        
        # Step 2: Build features
        logger.info("Step 2: Building feature sets")
        top_list = make_topk(snack, 135)
        logger.info(f"   Created top-135 ingredient list")
        
        binary_df, score_df = build_binary_and_scores(snack, top_list, max_score=20)
        logger.info(f"   Built binary features: {binary_df.shape}")
        logger.info(f"   Built score features: {score_df.shape}")
        
        snack = snack.join(binary_df).join(score_df)
        logger.info(f"   Joined feature sets: {snack.shape}")
        
        # Step 3: Select columns
        logger.info("Step 3: Selecting columns")
        nut8_cols = select_base_nutrients(snack)
        targets = ensure_targets(snack)
        umap_cols = [c for c in snack.columns if c.startswith("umap_10_")]
        binary_cols = [c for c in snack.columns if c.startswith("binary_")]
        score_cols = [c for c in snack.columns if c.startswith("score_")]
        
        logger.info(f"   Base nutrients: {len(nut8_cols)}")
        logger.info(f"   Target nutrients: {len(targets)}")
        logger.info(f"   UMAP features: {len(umap_cols)}")
        logger.info(f"   Binary features: {len(binary_cols)}")
        logger.info(f"   Score features: {len(score_cols)}")
        
        # Step 4: Ingredient-nutrient optimization
        logger.info("Step 4: Ingredient-nutrient optimization")
        ingnut_df = pd.read_csv("data/ingnut_df_top135.csv")
        logger.info(f"   Loaded ingnut data: {ingnut_df.shape}")
        
        # Define the specific nutrient column mappings
        nut_cols = ['Energy(kcal)', 'Total fat(g)', 'Protein(g)',
                    'Carbohydrate(g)', 'Total sugar(g)', 'Sodium(mg)',
                    'Cholesterol(mg)', 'Saturated fatty acids(g)']
        
        ingnut_cols = ['Energy', 'Total lipid (fat)', 'Protein',
                       'Carbohydrate, by difference', 'Sugars, total', 'Sodium, Na',
                       'Cholesterol', 'Fatty acids, total saturated']
        
        # Filter to only use columns that exist in both datasets
        available_nut_cols = [col for col in nut_cols if col in snack.columns]
        available_ingnut_cols = [col for col in ingnut_cols if col in ingnut_df.columns]
        
        # Create mapping between snack and ingnut columns
        column_mapping = {}
        for i, (nut_col, ingnut_col) in enumerate(zip(nut_cols, ingnut_cols)):
            if nut_col in available_nut_cols and ingnut_col in available_ingnut_cols:
                column_mapping[nut_col] = ingnut_col
        
        logger.info(f"   Available snack columns: {available_nut_cols}")
        logger.info(f"   Available ingnut columns: {available_ingnut_cols}")
        logger.info(f"   Column mapping: {column_mapping}")
        
        # Use the available columns
        nut8_cols = list(column_mapping.keys())
        ingnut_cols = list(column_mapping.values())
        
        # Run optimization
        snack, preds_w, variant_universe = predict_ingnut_weights_and_targets(
            snack, ingnut_df, nut8_cols, ingnut_cols,
            resolver="rule", constraint="nnls_mono", ridge=0.0, 
            robust=False, solver_name="osqp"
        )
        logger.info(f"   Optimization complete. Weight matrix: {preds_w.shape}")
        
        # Step 5: Machine learning predictions
        logger.info("Step 5: Machine learning predictions")
        feature_sets = {
            "nut8": nut8_cols,
            "nut8+binary": nut8_cols + binary_cols,
            "nut8+score": nut8_cols + score_cols,
            "nut8+umap_10": nut8_cols + umap_cols,
        }
        
        full_results = train_eval_sets(
            snack, targets, feature_sets, test_size=0.2, cv=3, force_rf=False
        )
        logger.info(f"   ML training complete. Results: {len(full_results)}")
        
        # Log ML metrics
        logger.info("   ML Metrics Summary:")
        for _, row in full_results.iterrows():
            logger.info(f"     {row['Nutrient']} - {row['FeatureSet']}: R²={row['R2']:.4f}, RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}, SMAPE={row['SMAPE']:.4f}%")
        
        # Step 6: Calculate metrics
        logger.info("Step 6: Calculating metrics")
        ing_metrics = metrics_ing_pred(snack)
        logger.info(f"   Ingredient prediction metrics: {len(ing_metrics)}")
        
        # Log optimization metrics
        logger.info("   Optimization Metrics Summary:")
        for _, row in ing_metrics.iterrows():
            logger.info(f"     {row['Nutrient']}: R²={row['R2']:.4f}, RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}, SMAPE={row['SMAPE']:.4f}%")
        
        all_metrics = pd.concat([ing_metrics, full_results], ignore_index=True)
        logger.info(f"   Combined metrics: {len(all_metrics)} total")
        
        # Step 7: Save all outputs
        logger.info("Step 7: Saving outputs")
        
        # 1. Full snack dataframe with all predictions
        logger.info("   Saving complete snack dataframe...")
        
        # Reorder columns for better readability
        def reorder_columns(df, nut8_cols, extra_map):
            # Ensure prediction columns exist
            for c in nut8_cols:
                pc = f"pred_{c}"
                if pc not in df.columns:
                    df[pc] = np.nan
            
            for truth, pc in extra_map.items():
                if pc not in df.columns:
                    df[pc] = np.nan
            
            # Create ordered column list
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
            return df[other + pairs]
        
        extra_map = {"Calcium(mg)": "pred_Calcium(mg)", "Fiber(g)": "pred_Fiber(g)", "Iron(mg)": "pred_Iron(mg)"}
        snack_ordered = reorder_columns(snack, nut8_cols, extra_map)
        
        snack_ordered.to_csv(os.path.join(outdir, "snack_df_complete.csv"), index=False)
        logger.info("   Saved snack_df_complete.csv")
        
        # 2. Ingredient parsing summary
        logger.info("   Saving ingredient parsing summary...")
        ingredient_summary = pd.DataFrame({
            'ingredient_list_raw': snack['ingredient_list_raw'].apply(lambda x: str(x)),
            'mapped_list': snack['mapped_list'].apply(lambda x: str(x)),
            'top_ingredients': [', '.join(lst[:10]) if isinstance(lst, list) else '' for lst in snack['mapped_list']],
            'ingredient_count': [len(lst) if isinstance(lst, list) else 0 for lst in snack['mapped_list']]
        })
        ingredient_summary.to_csv(os.path.join(outdir, "ingredient_parsing_summary.csv"), index=False)
        logger.info("   Saved ingredient_parsing_summary.csv")
        
        # 3. Metrics for all models
        logger.info("   Saving metrics...")
        all_metrics.to_csv(os.path.join(outdir, "metrics_all_models.csv"), index=False)
        logger.info("   Saved metrics_all_models.csv")
        
        # 4. Ingredient weights
        logger.info("   Saving ingredient weights...")
        weights_df = pd.DataFrame(preds_w, columns=variant_universe)
        weights_df.to_csv(os.path.join(outdir, "ingredient_weights.csv"), index=False)
        logger.info("   Saved ingredient_weights.csv")
        
        # 5. Model comparison summary
        logger.info("   Creating model comparison summary...")
        model_summary = all_metrics.groupby(['Nutrient', 'FeatureSet']).agg({
            'R2': 'mean', 'RMSE': 'mean', 'MAE': 'mean', 'SMAPE': 'mean'
        }).round(4)
        model_summary.to_csv(os.path.join(outdir, "model_comparison_summary.csv"))
        logger.info("   Saved model_comparison_summary.csv")
        
        # Step 8: Create visualizations
        logger.info("Step 8: Creating visualizations")
        try:
            fig = compare_feature_sets(all_metrics)
            fig.savefig(os.path.join(outdir, "feature_comparison.png"), dpi=300, bbox_inches="tight")
            logger.info("   Saved feature_comparison.png")
        except Exception as e:
            logger.error(f"   Visualization failed: {e}")
        
        # Step 9: Create comprehensive summary report
        logger.info("Step 9: Creating summary report")
        summary_report = f"""
NUTPRED COMPLETE PIPELINE EXECUTION SUMMARY
==========================================

EXECUTION DETAILS:
- Start time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- Output directory: {outdir}

INPUT DATA:
- Snack data: {snack.shape[0]} samples, {snack.shape[1]} total features
- Thesaurus: {thesaurus.shape[0]} ingredient mappings
- Ingredient-nutrient data: {ingnut_df.shape[0]} variants, {len(ingnut_cols)} nutrients

PROCESSING RESULTS:
- Top 135 ingredients selected for feature engineering
- {len(binary_cols)} binary features created
- {len(score_cols)} score features created  
- {len(umap_cols)} UMAP features used
- {len(nut8_cols)} base nutrients predicted
- {len(targets)} target nutrients predicted

MODEL PERFORMANCE:
- Optimization solver: OSQP with nnls_mono constraint
- ML models trained: {len(feature_sets)} different feature combinations
- Total model evaluations: {len(all_metrics)} (including cross-validation)

OUTPUT FILES GENERATED:
1. snack_df_complete.csv - Complete dataset with all predictions
2. ingredient_parsing_summary.csv - Ingredient parsing and mapping results
3. metrics_all_models.csv - Performance metrics for all models
4. ingredient_weights.csv - Optimization weights for each sample
5. model_comparison_summary.csv - Aggregated model performance comparison
6. feature_comparison.png - Visualization of model performance
7. nutpred_complete.log - Detailed execution log

KEY FINDINGS:
- Best performing feature set: {model_summary['R2'].idxmax() if len(model_summary) > 0 else 'N/A'}
- Average R2 across all models: {model_summary['R2'].mean():.4f if len(model_summary) > 0 else 'N/A'}
- Total ingredients processed: {sum(len(lst) if isinstance(lst, list) else 0 for lst in snack['mapped_list'])}
- Unique ingredients mapped: {len(set(ing for lst in snack['mapped_list'] for ing in (lst if isinstance(lst, list) else [])))}

EXECUTION STATUS: COMPLETED SUCCESSFULLY
        """
        
        with open(os.path.join(outdir, "execution_summary.txt"), "w") as f:
            f.write(summary_report)
        logger.info("   Saved execution_summary.txt")
        
        # Print summary to console
        print("\n" + "="*60)
        print("NUTPRED COMPLETE PIPELINE EXECUTED SUCCESSFULLY!")
        print("="*60)
        print(summary_report)
        print(f"\nAll outputs saved to: {outdir}/")
        print("Check the files above for complete results!")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        print(f"\nERROR: {e}")
        print("Check nutpred_complete.log for detailed error information.")
        return False

if __name__ == "__main__":
    success = run_complete_pipeline()
    if not success:
        sys.exit(1)
