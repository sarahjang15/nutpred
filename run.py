#!/usr/bin/env python3
"""
Consolidated nutpred pipeline runner with optional test mode
"""

import os
import sys
import logging
import traceback
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nutpred.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

def run_pipeline(test_size=None, random_state=42, filter_type="ingredients", filter_values=["popcorn", "pretzel", "pretzels"]):
    """Run the nutpred pipeline with optional test mode"""
    
    try:
        # Determine if this is a test run
        is_test_mode = test_size is not None
        mode_str = "TEST" if is_test_mode else "COMPLETE"
        
        logger.info(f"Starting nutpred {mode_str.lower()} pipeline")
        
        # Force reload all modules to ensure latest versions
        logger.info("Forcing module reload to ensure latest versions...")
        import importlib
        
        # List of modules to reload
        modules_to_reload = [
            'nutpred.preprocess',
            'nutpred.pred_by_ingnut', 
            'nutpred.pred_by_fullnut',
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
            filter_ingredients_to_topk, is_first_mapped, calculate_mapped_ratio,
            process_ingredients, ensure_umap_columns, select_base_nutrients, ensure_targets
        )
        from nutpred.pred_by_ingnut import predict_ingnut_weights_and_targets, eval_ing_pred
        from nutpred.pred_by_fullnut import train_tree_models
        from nutpred.viz import compare_feature_sets, create_scatterplots, create_shap_plots, create_filter_comparison_plots
        
        logger.info("Module reload complete - using latest versions")
        
        # Step 1: Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data")
        food_df, thesaurus_df = load_inputs(args.food_df, args.thesaurus_df)
        logger.info(f"   Loaded food data: {food_df.shape}")
        logger.info(f"   Loaded thesaurus: {thesaurus_df.shape}")
        
        # Apply test size limit if specified
        if is_test_mode:
            food_df = food_df.sample(test_size, random_state=random_state).reset_index(drop=True).copy()
            logger.info(f"   Using {len(food_df)} samples for testing")
        
        # Create ingredient mapping first (needed for filtering)
        food_df = ensure_mapped_list_column(food_df, thesaurus_df)
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
            base_outdir = "test_outputs" if is_test_mode else "complete_outputs"
        else:
            values_str = "_".join(filter_values)
            base_outdir = f"{'test' if is_test_mode else 'complete'}_outputs_{filter_type_str}_{values_str}"
        
        outdir = base_outdir
        os.makedirs(outdir, exist_ok=True)
        logger.info(f"Created output directory: {outdir}")
        
        # Ensure UMAP columns
        filtered_df = ensure_umap_columns(filtered_df, expected_dim=10)
        logger.info("   Ensured UMAP columns")
        
        # Select columns
        logger.info("   Selecting base nutrients and target nutrients")
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
        
        # Load ingnut data
        ingnut_df = pd.read_csv(args.ingnut_df)
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
        for food_col, ingnut_col in zip(nut8_cols, ingnut_cols):
            column_mapping[food_col] = ingnut_col
        
        logger.info(f"   Column mapping: {column_mapping}")
        
        # Create train/test split for evaluation (not for optimization training)
        first_mapped_mask = filtered_df.apply(is_first_mapped, axis=1)
        filtered_df["first_mapped"] = first_mapped_mask
        first_mapped_count = first_mapped_mask.sum()
        logger.info(f"   Samples passing first_mapped filter candidate for optimization: {first_mapped_count}/{len(filtered_df)}")

        strict_indices = filtered_df[first_mapped_mask].index.tolist()
        strict_mask = first_mapped_mask
        strict_count = strict_mask.sum()
        if strict_count == 0:
            raise ValueError(f"No samples pass first_mapped filter: {strict_count}")

        # Create strict_df for optimization
        strict_df = filtered_df[strict_mask].copy() 
        if len(strict_indices) < 10:
            logger.warning(f"   Not enough strict samples for train/test split: {len(strict_indices)}")
        
        # Create train/test split for evaluation
        train_idx, test_idx = train_test_split(range(len(strict_indices)), test_size=0.2, random_state=random_state)
        test_indices = [strict_indices[i] for i in test_idx]
        logger.info(f"   Test samples for evaluation: {len(test_indices)}")
        
        # Step 2: Run ingredient-nutrient optimization on first mapped samples
        logger.info("Step 2: Ingredient-nutrient optimization")
        logger.info("   Running ingredient-nutrient optimization on first mapped samples") 
        logger.info(f"   Optimization parameters: resolver={args.opt_solver}, constraint={args.opt_constraint}, ridge={args.opt_ridge}, robust={args.opt_robust}, solver={args.opt_solver}, scale={args.opt_scale}")
        
        # Run optimization accoding to the parameters
        strict_df, preds_w, failed_indices = predict_ingnut_weights_and_targets(
            strict_df, ingnut_df, nut8_cols, ingnut_cols, 
            constraint=args.opt_constraint, ridge=args.opt_ridge, robust=args.opt_robust, 
            solver_name=args.opt_solver, scale=args.opt_scale, test_indices=test_indices, top_list=top_list
        )
        logger.info(f"   Optimization complete. Weight matrix: {preds_w.shape}")
        
        # Track failed optimizations (samples with no ingredients or optimization failures)
        failed_set = set(failed_indices)
        for i, row in strict_df.iterrows():
            mapped_list = row.get('mapped_list', [])
            if not isinstance(mapped_list, list) or len(mapped_list) == 0:
                failed_set.add(i)
            # Check if all predictions are NaN (indicates optimization failure)
            pred_cols = [f"opt_{col}" for col in nut8_cols]
            if all(pd.isna(strict_df.loc[i, col]) for col in pred_cols if col in strict_df.columns):
                failed_set.add(i)
        
        strict_df["failed"] = strict_df.index.isin(failed_set).astype(int)
        logger.info(f"   Failed optimization samples: {len(failed_set)}/{len(strict_df)}")

        # Save optimization metrics
        logger.info("Calculating optimization metrics...")
        opt_metrics = eval_ing_pred(strict_df, failed_indices=failed_set, test_indices=test_indices, nut8_cols=nut8_cols)
        logger.info(f"Optimization metrics calculation complete.")
        
        # Step 3: Train and evaluate ML prediction models 
        logger.info("Step 3: Training and evaluating ML prediction models")
        feature_sets = {
            "nut8": nut8_cols,
            "nut8+binary": nut8_cols + binary_cols,
            "nut8+score": nut8_cols + score_cols,
            "nut8+umap_10": nut8_cols + umap_cols,
        }
        
        ml_metrics, models_dict, ml_test_indices, ml_output_df = train_tree_models(strict_df, feature_sets, targets, test_indices, cv=3)

        # Step 4: Save outputs
        logger.info("Step 4: Saving outputs")
        
        # Get first 3 columns from original snack data
        first_3_cols = filtered_df.columns[:3].tolist()
      
        # (1) Ingredient parsing summary
        logger.info("   (1) Saving ingredient parsing summary...")

        ingredient_summary = pd.DataFrame({
            # First 3 columns from original data
            **{col: filtered_df[col].values for col in first_3_cols},
            'ingredients': filtered_df['ingredients'],
            # Parsing related columns
            'umap_10': filtered_df['umap_10'].apply(lambda x: str(x)) if 'umap_10' in filtered_df.columns else [''] * len(filtered_df),
            'ingredient_list': filtered_df['ingredient_list_raw'],
            'ingredient_list_top20': filtered_df['ingredient_list_top20'],
            'mapped_ingredient_list': filtered_df['mapped_list'],
            'mapped_list_topk_only': filtered_df['mapped_list_topk_only'],
            'mapped_count': [len(lst) if isinstance(lst, list) else 0 for lst in filtered_df['mapped_list']],
            'mapped_topk_count': [len(lst) if isinstance(lst, list) else 0 for lst in filtered_df['mapped_list_topk_only']],
            'first_mapped': filtered_df['first_mapped'],
            'mapped_ratio': filtered_df.apply(calculate_mapped_ratio, axis=1).apply(round, args=(4,))
        })
        
        summary_filename = f"ingredient_parsing_{'test' if is_test_mode else ''}.csv"
        ingredient_summary.to_csv(os.path.join(outdir, summary_filename), index=False)
        logger.info(f"   Saved {summary_filename}")
        
        # (2) Ingredient weights
        logger.info("   (2) Saving ingredient weights...")
        weights_df = pd.DataFrame(preds_w, columns=top_list)
        
        # Add first 3 columns from original snack data
        for col in first_3_cols:
            weights_df[col] = filtered_df[first_mapped_mask][col].values
        
        # Reorder columns to put first 3 columns at the beginning
        weights_df = weights_df[first_3_cols + [col for col in weights_df.columns if col not in first_3_cols]]
        weights_filename = f"ingredient_weights_{'test' if is_test_mode else ''}.csv"
        weights_df.to_csv(os.path.join(outdir, weights_filename), index=False)
        logger.info(f"   Saved {weights_filename}")
        
         # (3) Saving optimization and ML predictions
        logger.info("   (3) Saving optimization and ML predictions...")
        all_metrics = pd.concat([opt_metrics, ml_metrics], ignore_index=True)
        if len(all_metrics) > 0:
            metrics_filename = f"metrics_{'test' if is_test_mode else 'all_models'}.csv"
            all_metrics.to_csv(os.path.join(outdir, metrics_filename), index=False)
            logger.info(f"   Saved {metrics_filename}")
        
        # (4) Model comparison summary
        logger.info("   (4) Creating model comparison summary...")
        if len(all_metrics) > 0 and 'Nutrient' in all_metrics.columns:
            logger.info("   Creating model comparison summary...")
            model_summary = all_metrics.groupby(['Nutrient', 'FeatureSet', 'Filter', 'SampleType']).agg({
                'R2': 'mean', 'RMSE': 'mean', 'MAE': 'mean', 'SMAPE': 'mean'
            }).round(4)
            model_summary_filename = f"model_comparison_{'test' if is_test_mode else 'summary'}.csv"
            model_summary.to_csv(os.path.join(outdir, model_summary_filename))
            logger.info(f"   Saved {model_summary_filename}")
        
         # (5) Add predictions to filtered_df
        logger.info("   (5) Adding predictions to filtered_df...")
        
        # Save sample info
        df_with_preds = filtered_df[first_3_cols + ['first_mapped']].copy()
        df_with_preds['SampleType'] = 'train'
        df_with_preds.loc[test_indices, 'SampleType'] = 'test'
        df_with_preds['Strict'] = 0

        for nutrient in nut8_cols:
            df_with_preds[nutrient] = filtered_df[nutrient].copy()
        
        # Add optimization predictions
        opt_pred_cols = [col for col in strict_df.columns if col.startswith('opt_')]
        df_with_preds = df_with_preds.merge(strict_df[first_3_cols + ['failed'] + opt_pred_cols], on=first_3_cols, how='left')
        
        # Update strict flag
        strict_mask = (df_with_preds['first_mapped'] == 1) & (df_with_preds['failed'] == 0)
        df_with_preds.loc[strict_mask, 'Strict'] = 1

        # Add ML predictions
        df_with_preds[ml_output_df.columns[3:]] = np.nan
        df_with_preds.loc[ml_output_df.index, ml_output_df.columns[3:]] = ml_output_df[ml_output_df.columns[3:]].copy()
        
        # Save predictions
        df_with_preds_filename = f"df_with_preds_{'test' if is_test_mode else 'all_models'}.csv"
        df_with_preds.to_csv(os.path.join(outdir, df_with_preds_filename), index=False)
        logger.info(f"   Saved {df_with_preds_filename}")

        raise ValueError("Stop here for now")
    
        # Step 6: Create visualizations
        logger.info("   (6) Creating visualizations...")
        try:    
            # Create feature comparison heatmaps
            compare_feature_sets(metrics_df=all_metrics)
            logger.info("   Created feature comparison heatmaps")
            
            # Create scatterplots
            create_scatterplots(filtered_df, outdir, metrics_df=all_metrics)
            logger.info("   Created scatterplots")
            
            # Create filter comparison plots
            create_filter_comparison_plots(filtered_df, outdir, metrics_df=all_metrics)
            logger.info("   Created filter comparison plots")
            
            # Create SHAP plots for XGBoost models (only if ML training succeeded)
            if models_dict:
                create_shap_plots(filtered_df, models_dict, feature_sets, outdir)
                logger.info("   Created SHAP plots")
        except Exception as e:
            logger.error(f"   Visualization failed: {e}")
            logger.error(traceback.format_exc())
        
        # Step 9: Create execution summary
        logger.info("Step 9: Creating execution summary")
        
        # Calculate statistics
        total_ingredients = sum(len(lst) if isinstance(lst, list) else 0 for lst in filtered_df['ingredient_list_raw'])
        unique_base_ingredients = len(set([ing for lst in filtered_df['ingredient_list_raw'] if isinstance(lst, list) for ing in lst]))
        unique_mapped_ingredients = len(set([ing for lst in filtered_df['mapped_list'] if isinstance(lst, list) for ing in lst]))
    
        
        summary_report = f"""
NUTPRED {mode_str} PIPELINE EXECUTION SUMMARY
{'=' * (len(mode_str) + 40)}

EXECUTION DETAILS:
- Start time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- Output directory: {outdir}
{test_size}- Filter type: {filter_type_str}
- Filter values: {filter_values}

INPUT DATA:
- Food data: {food_df.shape[0]} samples, {food_df.shape[1]} total features
- Filtered data: {filtered_df.shape[0]} samples, {filtered_df.shape[1]} total features
- Thesaurus: {thesaurus_df.shape[0]} ingredient mappings
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
1. {df_with_preds_filename} - Complete dataset with all predictions
2. {summary_filename} - Ingredient parsing and mapping results
3. {metrics_filename if len(all_metrics) > 0 else 'N/A'} - Performance metrics for all models
4. {weights_filename} - Optimization weights for each sample
5. {model_summary_filename if len(all_metrics) > 0 and 'Nutrient' in all_metrics.columns else 'N/A'} - Aggregated model performance comparison
6. feature_comparison.png - Visualization of model performance
7. plots/ - Directory containing scatterplots and SHAP plots (if ML training succeeded)
8. nutpred.log - Detailed execution log

EXECUTION STATUS: COMPLETED SUCCESSFULLY
        """
        
        summary_filename = f"execution_summary_{'test' if is_test_mode else ''}.txt"
        with open(os.path.join(outdir, summary_filename), "w") as f:
            f.write(summary_report)
        logger.info(f"   Saved {summary_filename}")
        
        # Print summary to console
        print("\n" + "="*60)
        print(f"NUTPRED {mode_str} PIPELINE EXECUTED SUCCESSFULLY!")
        print("="*60)
        print(summary_report)
        print(f"\nAll outputs saved to: {outdir}/")
        print("Check the files above for complete results!")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        print(f"\nERROR: {e}")
        print("Check nutpred.log for detailed error information.")
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict Calcium/Fiber/Iron from ingredients + nutrients.")

    # Test/complete mode
    ap.add_argument("--test-size", type=int, default=None, 
                       help="Number of samples to test with (if not specified, runs complete pipeline)")
    ap.add_argument("--random-state", type=int, default=42, 
                       help="Random state for reproducibility")

    # Filtering options
    ap.add_argument("--filter-type", choices=["ingredients", "category"], default="ingredients",
                       help="Type of filtering: 'ingredients' (default) or 'category'")
    ap.add_argument("--filter-values", nargs='+', 
                       default=["popcorn", "pretzel", "pretzels"], 
                       help="Values to filter by (ingredient keywords or category names)")

    # File paths
    ap.add_argument("--food-df", type=str, default="data/snack_input_df.csv")
    ap.add_argument("--thesaurus-df", type=str, default="data/THESAURUSFORPUBLICRELEASE.XLSX")
    ap.add_argument("--ingnut-df", type=str, default="data/ingnut_df_top135.csv")
    ap.add_argument("--outdir", type=str, default="./nutpred_outputs")

    # Tree-based prediction modeling
    ap.add_argument("--exclude", type=str, default="popcorn,pretzel,pretzels")
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--force-rf", action="store_true")

    # Optimization (ing_pred) - only rule-based resolver supported
    ap.add_argument("--opt-constraint", type=str, default="nnls_mono",
                    choices=["nnls_only", "nnls_mono", "eq1", "eq1_mono", "le1", "le1_mono"])
    ap.add_argument("--opt-solver", type=str, default="osqp",
                    choices=["osqp", "clarabel", "scs", "ecos", "piqp"])
    ap.add_argument("--opt-ridge", type=float, default=0.0)
    ap.add_argument("--opt-robust", action="store_true")
    ap.add_argument("--opt-scale", type=str, default="std",
                    choices=["std", "pminmax", "logiqr"],
                    help="Scaling mode for optimization: std (default), pminmax, logiqr")
    args = ap.parse_args()
    
    # Determine mode
    if args.test_size is not None:
        print(f"Running in TEST mode with {args.test_size} samples")
    else:
        print("Running in COMPLETE mode with full dataset")
    
    success = run_pipeline(
        test_size=args.test_size, 
        random_state=args.random_state, 
        filter_type=args.filter_type, 
        filter_values=args.filter_values
    )
    
    if not success:
        sys.exit(1)
