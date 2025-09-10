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
import yaml
from datetime import datetime

# Make logs directory
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/nutpred_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'),
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
            load_inputs, make_id_col, ensure_mapped_list_column, filter_rows, filter_by_category, filter_by_ingredients, extract_topk_from_ingnut,
            filter_ingredients_to_topk, is_first_mapped, calculate_mapped_ratio, calculate_mapped_ratio_top20,
            preprocess_pipeline, ensure_umap_columns, select_base_nutrients, ensure_targets
        )
        from nutpred.pred_by_ingnut import predict_ingnut_weights_and_targets, eval_ing_pred
        from nutpred.pred_by_fullnut import train_tree_models
        from nutpred.viz import compare_feature_sets, create_scatterplots, create_shap_plots
        
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

        # Make id column
        food_df = make_id_col(food_df)
        logger.info("   Created id column")
        
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
        
        values_str = "_".join(filter_values)

        base_outdir = f"{args.outdir}_test" if is_test_mode else f"{args.outdir}_complete"
        
        outdir = base_outdir
        os.makedirs(outdir, exist_ok=True)
        logger.info(f"Created output directory: {outdir}")

        # Store config
        config = {
        "filter_type": args.filter_type,
        "filter_values": args.filter_values,
        "test_size": args.test_size,
        "outdir": args.outdir,
        "base_outdir": base_outdir,
        "test_size": args.test_size,
        "is_test_mode": is_test_mode,
        "random_state": args.random_state,
        "food_df": args.food_df,
        "thesaurus_df": args.thesaurus_df,
        "ingnut_df": args.ingnut_df,
        "outdir": args.outdir,
        "cv": args.cv,
        "force_rf": args.force_rf,
        "opt_constraint": args.opt_constraint,
        "opt_solver": args.opt_solver,
        "opt_ridge": args.opt_ridge,
        "opt_robust": args.opt_robust,
        "opt_scale": args.opt_scale
    }

        config_path = os.path.join(base_outdir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"   Saved config: {config_path}")
        
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
        
        # Preprocess food df and create features
        filtered_df = preprocess_pipeline(filtered_df, ingnut_df, k=133)
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
        
        logger.info(f"   Available nut8 columns: {nut8_cols}")
        logger.info(f"   Available ingnut columns: {ingnut_cols}")
        
        # Create mapping between snack and ingnut column names
        column_mapping = {}
        for food_col, ingnut_col in zip(nut8_cols, ingnut_cols):
            column_mapping[food_col] = ingnut_col
        
        logger.info(f"   Column mapping: {column_mapping}")
        
        # Create train/test split for evaluation (not for optimization training)
        train_idx, test_idx = train_test_split(range(len(filtered_df)), test_size=0.2, random_state=random_state)
        logger.info(f"   Created train/test split: {len(train_idx)} train samples, {len(test_idx)} test samples")
        
        filtered_df["SampleType"] = "train"
        filtered_df.loc[test_idx, "SampleType"] = "test"
        
        # Make df for each filter group
        group_cols = ["first_mapped", "mapped_ratio_high", "mapped_ratio_top20_high", "strict"]
        group_dfs = {}
        for group_col in group_cols:
            group_dfs[group_col] = filtered_df[filtered_df[group_col]].copy()
            logger.info(f"   Created group dataframe for {group_col}: {group_dfs[group_col].shape}")
        group_dfs["full"] = filtered_df
        group_cols.append("full")
        logger.info(f"   Full dataframe size is: {group_dfs['full'].shape}")


         # Step 2: Run ingredient-nutrient optimization 
        logger.info("Step 2: Ingredient-nutrient optimization")
        logger.info("   Running ingredient-nutrient optimization for each filter groups") 
        logger.info(f"   Optimization parameters: resolver={args.opt_solver}, constraint={args.opt_constraint}, ridge={args.opt_ridge}, robust={args.opt_robust}, solver={args.opt_solver}, scale={args.opt_scale}")

        opt_failed_indices = {}
        opt_preds_w = {}
        for group_name, group_df in group_dfs.items():
            group_df, preds_w, failed_indices = predict_ingnut_weights_and_targets(
                group_df, ingnut_df, nut8_cols, ingnut_cols, 
            constraint=args.opt_constraint, ridge=args.opt_ridge, robust=args.opt_robust, 
            solver_name=args.opt_solver, scale=args.opt_scale, top_list=top_list, group_name=group_name
            )
            opt_failed_indices[group_name], opt_preds_w[group_name] = failed_indices, preds_w
            group_df["failed"] = False
            # Filter failed_indices to only include indices that exist in group_df
            valid_failed_indices = [idx for idx in failed_indices if idx in group_df.index]
            if valid_failed_indices:
                group_df.loc[valid_failed_indices, "failed"] = True
            logger.info(f"   {group_name}: {len(valid_failed_indices)}/{len(failed_indices)} failed indices are valid in this group")   
            group_dfs[group_name] = group_df
            logger.info(f"   Optimization complete for {group_name}. Weight matrix: {opt_preds_w[group_name].shape}")
        logger.info(f"Optimization complete for all groups.")

        # Save optimization metrics
        optimization_metrics = {}
        logger.info("Calculating optimization metrics...")
        for group_name, group_df in group_dfs.items():
            optimization_metrics[group_name] = eval_ing_pred(group_df, nut8_cols=nut8_cols, group_name=group_name)
            logger.info(f"  {group_name} optimization metrics calculated.")
        logger.info(f"Optimization metrics calculation complete for all groups.")
        
        # Step 3: Train and evaluate ML prediction models 
        logger.info("Step 3: Training and evaluating ML prediction models")
        feature_sets = {
            "nut8": nut8_cols,
            "nut8+binary": nut8_cols + binary_cols,
            "nut8+score": nut8_cols + score_cols,
            "nut8+umap_10": nut8_cols + umap_cols,
        }
        
        ml_metrics = {}
        ml_output_dfs = {}
        models_dicts = {}
        for group_name, group_df in group_dfs.items():
            ml_metrics[group_name], models_dicts[group_name], ml_output_dfs[group_name] = train_tree_models(group_df, feature_sets, targets, test_idx, cv=3, group_name=group_name)

        # Step 4: Save outputs
        logger.info("Step 4: Saving outputs")
        
        # Get id columns
        id_cols = ['id_col'] + filtered_df.columns[:3].tolist()
      
        # (1) Ingredient parsing summary
        logger.info("   (1) Saving ingredient parsing summary...")

        ingredient_summary = pd.DataFrame({
            **{col: filtered_df[col].values for col in id_cols},
            'ingredients': filtered_df['ingredients'],
            'umap_10': filtered_df['umap_10'].apply(lambda x: str(x)) if 'umap_10' in filtered_df.columns else [''] * len(filtered_df),
            'ingredient_list': filtered_df['ingredient_list_raw'],
            'ingredient_list_top20': filtered_df['ingredient_list_top20'],
            'mapped_ingredient_list': filtered_df['mapped_list'],
            'mapped_list_topk_only': filtered_df['mapped_list_topk_only'],
            'mapped_count': [len(lst) if isinstance(lst, list) else 0 for lst in filtered_df['mapped_list']],
            'mapped_topk_count': [len(lst) if isinstance(lst, list) else 0 for lst in filtered_df['mapped_list_topk_only']],
            'mapped_ratio': filtered_df['mapped_ratio'],
            'mapped_ratio_top20': filtered_df['mapped_ratio_top20'],
            **{col: filtered_df[col].values for col in [col for col in group_cols if col != "full"]},
            **{col: filtered_df[col].values for col in binary_cols},
            **{col: filtered_df[col].values for col in score_cols}
        })
        
        summary_filename = f"ingredient_parsing_{'test' if is_test_mode else ''}.csv"
        ingredient_summary.to_csv(os.path.join(outdir, summary_filename), index=False)
        logger.info(f"   Saved {summary_filename}")
        
        # (2) Ingredient weights
        logger.info("   (2) Saving ingredient weights for first_mapped dataset...")
        weights_df = pd.DataFrame(opt_preds_w["first_mapped"], columns=top_list)
        
        # Add id and group columns
        for col in id_cols:
            weights_df[col] = filtered_df[filtered_df["first_mapped"]][col]
        
        # Reorder columns to put id columns at the beginning
        weights_df = weights_df[id_cols + [col for col in weights_df.columns if col not in id_cols]]
        weights_filename = f"ingredient_weights_{'test' if is_test_mode else ''}_full.csv"
        weights_df.to_csv(os.path.join(outdir, weights_filename), index=False)
        logger.info(f"   Saved {weights_filename}")
        
         # (3) Saving optimization and ML predictions
        logger.info("   (3) Saving optimization and ML predictions...")

        all_metrics = pd.DataFrame()
        for group_name, group_df in group_dfs.items():
            metrics = pd.concat([optimization_metrics[group_name], ml_metrics[group_name]], ignore_index=True)
            metrics["Group"] = group_name
            if len(metrics) ==  0:
                continue
            all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)
        metrics_filename = f"metrics_{'test' if is_test_mode else 'all_models'}.csv"
        all_metrics.to_csv(os.path.join(outdir, metrics_filename), index=False)
        logger.info(f"   Saved {metrics_filename}")
        

         # (4) Add predictions to filtered_df
        logger.info("   (5) Adding predictions to filtered_df...")
        
        # Save sample info
        df_with_preds = filtered_df[id_cols + [col for col in group_cols if col != "full"]].copy()
        df_with_preds['SampleType'] = 'train'
        # Filter test_idx to only include indices that exist in df_with_preds
        valid_test_idx = [idx for idx in test_idx if idx in df_with_preds.index]
        if valid_test_idx:
            df_with_preds.loc[valid_test_idx, 'SampleType'] = 'test'
      
        for nutrient in nut8_cols:
            df_with_preds[nutrient] = filtered_df[nutrient].copy()
        
        # Add optimization predictions
        for group_name, _df in group_dfs.items():
            opt_pred_cols = [col for col in _df.columns if '_opt_' in col]
            df_with_preds = df_with_preds.merge(_df[id_cols + opt_pred_cols], on=id_cols, how='left')
    
        # Add ML predictions
        for group_name, ml_output_df in ml_output_dfs.items():

            ml_pred_cols = [col for col in ml_output_df.columns if '_xgb_' in col]
            df_with_preds = df_with_preds.merge(ml_output_df[id_cols + ml_pred_cols], on=id_cols, how='left')

        # Reorder columns for easy comparison
        ordered_cols = id_cols + [col for col in group_cols if col != "full"] 
        ordered_cols += ['SampleType']
        for nutrient in nut8_cols + targets:
            ordered_cols += [col for col in df_with_preds.columns if nutrient in col] 

        df_with_preds = df_with_preds[ordered_cols]

        # Save predictions
        df_with_preds_filename = f"df_with_preds_{'test' if is_test_mode else 'all_models'}.csv"
        df_with_preds.to_csv(os.path.join(outdir, df_with_preds_filename), index=False)
        logger.info(f"   Saved {df_with_preds_filename}")

        #raise ValueError("Stop here for now")
    
        # Step 6: Create visualizations
        logger.info("   (6) Creating visualizations...")
        # Create plots directory
        plots_dir = os.path.join(outdir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        try:    
            # Create model comparison heatmaps
            compare_feature_sets(metrics_df=all_metrics, metrics_file_path=outdir+'/'+metrics_filename, outdir=plots_dir)
            logger.info("   Created model comparison heatmaps")
            
            # Create scatterplots
            create_scatterplots(pred_file_path=outdir+'/'+df_with_preds_filename, outdir=plots_dir)
            logger.info("   Created scatterplots")
            
            # Create SHAP plots for XGBoost models (only if ML training succeeded)
            #create_shap_plots(filtered_df, models_dicts, feature_sets, outdir=plots_dir)
            #logger.info("   Created SHAP plots")
            logger.info("   Skipping SHAP plots for now")
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

5. plots/ - Directory containing comprehensive visualizations:
   - metrics_comparison_heatmap.png - Model performance comparison heatmap
   - scatterplots/ - True vs predicted scatterplots for all feature sets and groups
   - shap/ - SHAP analysis plots for XGBoost models (if available)
6. nutpred.log - Detailed execution log

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
