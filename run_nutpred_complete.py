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
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from nutpred.metrics import smape

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

def run_complete_pipeline(filter_type="ingredients", filter_values=["popcorn", "pretzel", "pretzels"]):
    """Run the complete nutpred pipeline with all outputs"""
    
    try:
        logger.info("Starting complete nutpred pipeline")
        
        # Force reload all modules to ensure latest versions
        logger.info("Forcing module reload to ensure latest versions...")
        import importlib
        import sys
        
        # List of modules to reload
        modules_to_reload = [
            'nutpred.preprocess',
            'nutpred.pred_by_ingnut', 
            'nutpred.metrics',
            'nutpred.viz'
        ]
        
        for module_name in modules_to_reload:
            if module_name in sys.modules:
                logger.info(f"Reloading module: {module_name}")
                importlib.reload(sys.modules[module_name])
            else:
                logger.info(f"Loading module: {module_name}")
        
        # Import all modules after reload
        from nutpred.preprocess import (
            load_inputs, ensure_mapped_list_column, filter_rows, filter_by_category, filter_by_ingredients, extract_topk_from_ingnut,
            filter_ingredients_to_topk, process_ingredients,
            is_first_mapped, calculate_mapped_ratio,
            ensure_umap_columns, select_base_nutrients, ensure_targets
        )
        from nutpred.pred_by_ingnut import predict_ingnut_weights_and_targets, metrics_ing_pred
        from nutpred.pred_by_fullnut import train_tree_models
        from nutpred.viz import compare_feature_sets, create_scatterplots, create_shap_plots, create_filter_comparison_plots
        
        logger.info("Module reload complete - using latest versions")
        
        # Step 1: Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data")
        food_df, thesaurus = load_inputs("data/snack_input_df.csv", "data/THESAURUSFORPUBLICRELEASE.XLSX")
        logger.info(f"   Loaded food data: {food_df.shape}")
        logger.info(f"   Loaded thesaurus: {thesaurus.shape}")
        
        # Create ingredient mapping first (needed for filtering)
        food_df = ensure_mapped_list_column(food_df, thesaurus)
        logger.info("   Created mapped ingredient lists")
        
        # Apply filtering based on filter type
        if filter_type == "category":
            # Use category-based filtering
            filtered_df = filter_by_category(food_df, filter_values)
            filter_type_str = "category"
        else:
            # Use ingredient-based filtering (exclude rows with these ingredients)
            filtered_df = filter_rows(food_df, exclude_terms=filter_values)
            filter_type_str = "ingredients"
        
        logger.info(f"   Filtered data using {filter_type_str}: {filtered_df.shape}")
        
        # Create output directory after filter_type_str is defined
        if filter_values == ["full"]:
            outdir = "complete_outputs"
        else:
            values_str = "_".join(filter_values)
            outdir = f"complete_outputs_{filter_type_str}_{values_str}"
        
        os.makedirs(outdir, exist_ok=True)
        logger.info(f"Created output directory: {outdir}")
        
        # Ensure UMAP columns
        filtered_df = ensure_umap_columns(filtered_df, expected_dim=10)
        logger.info("   Ensured UMAP columns")
        
        # Step 2: Build features (will be done after loading ingnut_df)
        logger.info("Step 2: Feature building will be done after loading ingnut_df")
        
        # Step 3: Select columns
        logger.info("Step 3: Selecting columns")
        nut8_cols = select_base_nutrients(filtered_df)
        targets = ensure_targets(filtered_df)
        umap_cols = [c for c in filtered_df.columns if c.startswith("umap_10_")]
        binary_cols = [c for c in filtered_df.columns if c.startswith("binary_")]
        score_cols = [c for c in filtered_df.columns if c.startswith("score_")]
        
        logger.info(f"   Base nutrients: {len(nut8_cols)}")
        logger.info(f"   Target nutrients: {len(targets)}")
        logger.info(f"   UMAP features: {len(umap_cols)}")
        logger.info(f"   Binary features: {len(binary_cols)}")
        logger.info(f"   Score features: {len(score_cols)}")
        
        # Step 4: Ingredient-nutrient optimization
        logger.info("Step 4: Ingredient-nutrient optimization")
        ingnut_df = pd.read_csv("data/ingnut_df_top135.csv")
        logger.info(f"   Loaded ingnut data: {ingnut_df.shape}")
        
        # Get the top-k list used for feature engineering
        ingnut_df, top_list = extract_topk_from_ingnut(ingnut_df, k=133)
        logger.info(f"   Using top-{len(top_list)} ingredients for optimization")
        

        # Process ingredients with butter resolution and create features
        filtered_df = process_ingredients(filtered_df, ingnut_df, k=133)
        logger.info(f"   Processed ingredients: {filtered_df.shape}")
        
        # Get feature columns
        binary_cols = [c for c in filtered_df.columns if c.startswith("binary_")]
        score_cols = [c for c in filtered_df.columns if c.startswith("score_")]
        umap_cols = [c for c in filtered_df.columns if c.startswith("umap_10_")]
        
        logger.info(f"   Binary features: {len(binary_cols)}")
        logger.info(f"   Score features: {len(score_cols)}")
        logger.info(f"   UMAP features: {len(umap_cols)}")
        
        # Map nutrient columns between snack and ingnut dataframes
        ingnut_cols = ['Energy', 'Total lipid (fat)', 'Protein', 'Carbohydrate, by difference', 
                      'Sugars, total', 'Sodium, Na', 'Cholesterol', 'Fatty acids, total saturated']
        
        logger.info(f"   Available snack columns: {nut8_cols}")
        logger.info(f"   Available ingnut columns: {ingnut_cols}")
        
        # Create mapping between snack and ingnut column names
        column_mapping = {}
        for snack_col, ingnut_col in zip(nut8_cols, ingnut_cols):
            column_mapping[snack_col] = ingnut_col
        
        logger.info(f"   Column mapping: {column_mapping}")
        
        # Create train/test split for evaluation (not for optimization training)
        first_mapped_mask = filtered_df.apply(is_first_mapped, axis=1)
        if "failed" in filtered_df.columns:
            failed_mask = filtered_df["failed"] == 0
            strict_mask = first_mapped_mask & failed_mask
        else:
            strict_mask = first_mapped_mask
        
        # Get strict samples indices for evaluation
        strict_indices = filtered_df[strict_mask].index.tolist()
        logger.info(f"   Strict samples for evaluation: {len(strict_indices)}")
        
        if len(strict_indices) < 10:
            logger.warning(f"   Not enough strict samples for train/test split: {len(strict_indices)}")
            # Use all samples for evaluation
            test_indices = None
        else:
            # Create train/test split on strict samples for evaluation
            train_idx, test_idx = train_test_split(
                range(len(strict_indices)), 
                test_size=0.2, 
                random_state=42
            )
            test_indices = [strict_indices[i] for i in test_idx]
            logger.info(f"   Evaluation train/test split: {len(train_idx)} train, {len(test_idx)} test")
        
        # Run optimization with actual ingredient names
        filtered_df, preds_w, failed_indices = predict_ingnut_weights_and_targets(
            filtered_df, ingnut_df, nut8_cols, ingnut_cols, 
            resolver="rule", constraint="nnls_mono", ridge=0.0, robust=False, 
            solver_name="osqp", scale="std", test_indices=test_indices, top_list=top_list
        )
        logger.info(f"   Optimization complete. Weight matrix: {preds_w.shape}")
        
        # Track failed optimizations (samples with no ingredients or optimization failures)
        failed_set = set(failed_indices)
        for i, row in filtered_df.iterrows():
            mapped_list = row.get('mapped_list', [])
            if not isinstance(mapped_list, list) or len(mapped_list) == 0:
                failed_set.add(i)
            # Check if all predictions are NaN (indicates optimization failure)
            pred_cols = [f"opt_{col}" for col in nut8_cols]
            if all(pd.isna(filtered_df.loc[i, col]) for col in pred_cols if col in filtered_df.columns):
                failed_set.add(i)
        
        filtered_df["failed"] = filtered_df.index.isin(failed_set).astype(int)
        logger.info(f"   Failed optimization samples: {len(failed_set)}/{len(filtered_df)}")
        
        # Step 5: Calculate all metrics (optimization and ML)
        logger.info("Step 5: Calculating all metrics")
        feature_sets = {
            "nut8": nut8_cols,
            "nut8+binary": nut8_cols + binary_cols,
            "nut8+score": nut8_cols + score_cols,
            "nut8+umap_10": nut8_cols + umap_cols,
        }
        
        # Calculate optimization metrics
        logger.info("Calculating optimization metrics...")
        opt_metrics = metrics_ing_pred(filtered_df, failed_indices=failed_set, test_indices=test_indices, nut8_cols=nut8_cols)
        logger.info(f"Optimization metrics calculated: {len(opt_metrics)} rows")
        
        # Calculate ML metrics
        logger.info("Calculating ML metrics...")
        ml_metrics, models_dict = train_tree_models(filtered_df, feature_sets, targets, test_size=0.2, cv=3)
        logger.info(f"ML metrics calculated: {len(ml_metrics)} rows")
        
        # Combine all metrics - handle empty DataFrames
        all_metrics_list = []
        if len(opt_metrics) > 0:
            all_metrics_list.append(opt_metrics)
        if len(ml_metrics) > 0:
            all_metrics_list.append(ml_metrics)
        
        if all_metrics_list:
            all_metrics = pd.concat(all_metrics_list, ignore_index=True)
        else:
            # Create empty DataFrame with expected columns
            all_metrics = pd.DataFrame(columns=['Nutrient', 'FeatureSet', 'Model', 'Filter', 'R2', 'RMSE', 'MAE', 'SMAPE'])
        
        logger.info(f"All metrics combined: {len(all_metrics)} total")
        
        # Log metrics summary
        logger.info("   Metrics Summary:")
        if len(all_metrics) > 0:
            for _, row in all_metrics.iterrows():
                logger.info(f"     {row['Nutrient']} - {row['FeatureSet']} ({row['Filter']}): R²={row['R2']:.4f}, RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}, SMAPE={row['SMAPE']:.4f}%")
        else:
            logger.warning("   No metrics available to display")
        
        # Step 6: Save all outputs
        logger.info("Step 6: Saving outputs")
        
        # 1. Full snack dataframe with all predictions
        logger.info("   Saving complete snack dataframe...")
        
        # Remove UMAP columns from output
        umap_cols_to_remove = [c for c in filtered_df.columns if c.startswith("umap_")]
        snack_clean = filtered_df.drop(columns=umap_cols_to_remove)
        logger.info(f"   Removed {len(umap_cols_to_remove)} UMAP columns from output")
        
        # Add predicted weights as a list column
        def create_weights_list(row_idx):
            if row_idx < len(preds_w):
                weights = preds_w[row_idx]
                # Convert to list and round to 4 decimal places
                return str([round(w, 4) for w in weights])
            return "[]"
        
        snack_clean["predicted_weights"] = [create_weights_list(i) for i in range(len(snack_clean))]
        
        # Reorder columns for better readability
        def reorder_columns(df, nut8_cols, extra_map):
            # Ensure prediction columns exist
            for c in nut8_cols:
                pc = f"opt_{c}"
                if pc not in df.columns:
                    df[pc] = np.nan
            
            for truth, pc in extra_map.items():
                if pc not in df.columns:
                    df[pc] = np.nan
            
            # Create ordered column list
            pairs = []
            for c in nut8_cols:
                pairs += [c, f"opt_{c}"]
            for truth, pc in extra_map.items():
                if truth in df.columns:
                    pairs += [truth, pc]
                else:
                    pairs += [pc]
            
            pair_set = set(pairs)
            other = [c for c in df.columns if c not in pair_set]
            return df[other + pairs]
        
        extra_map = {"Calcium(mg)": "opt_Calcium(mg)", "Fiber(g)": "opt_Fiber(g)", "Iron(mg)": "opt_Iron(mg)"}
        snack_ordered = reorder_columns(snack_clean, nut8_cols, extra_map)
        
        # Dynamic file name based on filter type and values
        if filter_values == ["full"]:
            output_filename = "snack_df_complete.csv"
        else:
            values_str = "_".join(filter_values)
            output_filename = f"food_df_{filter_type_str}_{values_str}_complete.csv"
        
        snack_ordered.to_csv(os.path.join(outdir, output_filename), index=False)
        logger.info(f"   Saved {output_filename}")
        
        # 2. Ingredient parsing summary (include first 3 columns and all parsing related columns)
        logger.info("   Saving ingredient parsing summary...")
        
        # Get first 3 columns from original snack data
        first_3_cols = filtered_df.columns[:3].tolist()
        
        # Create first_mapped column using imported function
        first_mapped_col = filtered_df.apply(is_first_mapped, axis=1)
        
        # Calculate mapped ratio using imported function
        mapped_ratio_col = filtered_df.apply(calculate_mapped_ratio, axis=1)
        
        ingredient_summary = pd.DataFrame({
            # First 3 columns from original data
            **{col: filtered_df[col].values for col in first_3_cols},
            'ingredients': filtered_df['ingredients'],
            # Parsing related columns
            'umap_10': filtered_df['umap_10'].apply(lambda x: str(x)) if 'umap_10' in filtered_df.columns else [''] * len(filtered_df),
            'ingredient_list': filtered_df['ingredient_list_raw'].apply(lambda x: str(x)),
            'ingredient_list_top20': filtered_df['ingredient_list_top20'].apply(lambda x: str(x)),
            'mapped_ingredient_list': filtered_df['mapped_list'].apply(lambda x: str(x)),
            'mapped_list_topk_only': filtered_df['mapped_list_topk_only'].apply(lambda x: str(x)),
            'mapped_count': [len(lst) if isinstance(lst, list) else 0 for lst in filtered_df['mapped_list']],
            'mapped_topk_count': [len(lst) if isinstance(lst, list) else 0 for lst in filtered_df['mapped_list_topk_only']],
            'first_mapped': first_mapped_col,
            'mapped_ratio': mapped_ratio_col
        })
        ingredient_summary.to_csv(os.path.join(outdir, "ingredient_parsing_summary.csv"), index=False)
        logger.info("   Saved ingredient_parsing_summary.csv")
        
        # 3. Metrics for all models
        logger.info("   Saving metrics...")
        all_metrics.to_csv(os.path.join(outdir, "metrics_all_models.csv"), index=False)
        logger.info("   Saved metrics_all_models.csv")
        
        # 4. Ingredient weights (include first 3 columns from snack_input_df)
        logger.info("   Saving ingredient weights...")
        weights_df = pd.DataFrame(preds_w, columns=top_list)
        
        # Add first 3 columns from original snack data
        first_3_cols = filtered_df.columns[:3].tolist()
        for col in first_3_cols:
            weights_df[col] = filtered_df[col].values
        
        # Reorder columns to put first 3 columns at the beginning
        weights_df = weights_df[first_3_cols + [col for col in weights_df.columns if col not in first_3_cols]]
        weights_df.to_csv(os.path.join(outdir, "ingredient_weights.csv"), index=False)
        logger.info("   Saved ingredient_weights.csv")
        
        # 5. Model comparison summary
        logger.info("   Creating model comparison summary...")
        if len(all_metrics) > 0 and 'Nutrient' in all_metrics.columns:
            model_summary = all_metrics.groupby(['Nutrient', 'FeatureSet', 'Filter']).agg({
                'R2': 'mean', 'RMSE': 'mean', 'MAE': 'mean', 'SMAPE': 'mean'
            }).round(4)
        else:
            logger.warning("   No metrics available for model comparison summary")
            model_summary = pd.DataFrame()
        
        if len(model_summary) > 0:
            model_summary.to_csv(os.path.join(outdir, "model_comparison_summary.csv"))
            logger.info("   Saved model_comparison_summary.csv")
        else:
            logger.warning("   No model comparison data to save")
        
        # Step 8: Create visualizations
        try:
            logger.info("   Creating visualizations...")
            
            # Create feature comparison heatmaps
            compare_feature_sets(metrics_df=all_metrics)
            logger.info("   Created feature comparison heatmaps")
            
            # Create scatterplots (only test samples for optimization)
            create_scatterplots(filtered_df, outdir, metrics_df=all_metrics)
            logger.info("   Created scatterplots")
            
            # Create filter comparison plots
            create_filter_comparison_plots(filtered_df, outdir, metrics_df=all_metrics)
            logger.info("   Created filter comparison plots")
            
            # Create SHAP plots for XGBoost models (only if ML training succeeded)
            create_shap_plots(filtered_df, models_dict, feature_sets, outdir)
            logger.info("   Created SHAP plots")
        except Exception as e:
            logger.error(f"   Visualization failed: {e}")
            logger.error(traceback.format_exc())
        
        # Step 9: Create comprehensive summary report
        logger.info("Step 9: Creating summary report")
        
        # Handle best performing feature set (MultiIndex tuple)
        best_feature_set = "N/A"
        avg_r2 = "N/A"
        total_ingredients = sum(len(lst) if isinstance(lst, list) else 0 for lst in filtered_df['mapped_list'])
        # Count unique ingredients
        # Use the top-k list used for feature engineering instead of counting all ingredients from all samples
        if 'top_list' in locals() and top_list is not None:
            unique_mapped_ingredients = len(top_list)  # This should be 133
            
            # Count unique base ingredients (before variant resolution) from the top-k list
            base_ingredients = []
            for ing in top_list:
                base_ing = ing.split(' | ')[0] if ' | ' in ing else ing
                base_ingredients.append(base_ing)
            unique_base_ingredients = len(set(base_ingredients))
        
        if len(model_summary) > 0:
            best_idx = model_summary['R2'].idxmax()
            if isinstance(best_idx, tuple):
                # Handle MultiIndex with Filter column
                if len(best_idx) == 3:
                    best_feature_set = f"{str(best_idx[0])} - {str(best_idx[1])} ({str(best_idx[2])})"
                else:
                    best_feature_set = f"{str(best_idx[0])} - {str(best_idx[1])}"
            else:
                best_feature_set = str(best_idx)
            avg_r2 = f"{model_summary['R2'].mean():.4f}"
        elif len(all_metrics) == 0: # Changed from full_results to all_metrics
            logger.warning("No ML results available for summary")
        
        summary_report = f"""
NUTPRED COMPLETE PIPELINE EXECUTION SUMMARY
==========================================

EXECUTION DETAILS:
- Start time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- Output directory: {outdir}
- Filter type: {filter_type_str}
- Filter values: {filter_values}

INPUT DATA:
- Food data: {food_df.shape[0]} samples, {food_df.shape[1]} total features
- Filtered data: {filtered_df.shape[0]} samples, {filtered_df.shape[1]} total features
- Thesaurus: {thesaurus.shape[0]} ingredient mappings
- Ingredient-nutrient data: {ingnut_df.shape[0]} ingredients, {len(ingnut_cols)} nutrients

PROCESSING RESULTS:
- Top k ingredients selected for feature engineering
- {len(binary_cols)} binary features created
- {len(score_cols)} score features created  
- {len(umap_cols)} UMAP features used
- {len(nut8_cols)} base nutrients predicted
- {len(targets)} target nutrients predicted

MODEL PERFORMANCE:
- Optimization solver: OSQP with nnls_mono constraint
- ML models trained: {len(all_metrics) // (len(targets) * 2) if len(all_metrics) > 0 else 0} different feature combinations (evaluated on strict and all samples)
- Total model evaluations: {len(all_metrics)} (including cross-validation and filter breakdowns)

OUTPUT FILES GENERATED:
1. snack_df_complete.csv - Complete dataset with all predictions
2. ingredient_parsing_summary.csv - Ingredient parsing and mapping results
3. metrics_all_models.csv - Performance metrics for all models
4. ingredient_weights.csv - Optimization weights for each sample
5. model_comparison_summary.csv - Aggregated model performance comparison
6. feature_comparison.png - Visualization of model performance
7. plots/ - Directory containing scatterplots and SHAP plots (if ML training succeeded)
8. nutpred_complete.log - Detailed execution log

KEY FINDINGS:
- Best performing feature set: {best_feature_set}
- Average R2 across all models: {avg_r2}
- Total ingredients processed: {total_ingredients}
- Unique base ingredients: {unique_base_ingredients}
- Unique mapped ingredients: {unique_mapped_ingredients}

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
    import argparse
    parser = argparse.ArgumentParser(description="Run nutpred complete pipeline")
    parser.add_argument("--filter-type", choices=["ingredients", "category"], default="ingredients",
                       help="Type of filtering: 'ingredients' (default) or 'category'")
    parser.add_argument("--filter-values", nargs='+', 
                       default=["popcorn", "pretzel", "pretzels"], 
                       help="Values to filter by (ingredient keywords or category names)")
    
    args = parser.parse_args()
    
    success = run_complete_pipeline(filter_type=args.filter_type, filter_values=args.filter_values)
    if not success:
        sys.exit(1)

