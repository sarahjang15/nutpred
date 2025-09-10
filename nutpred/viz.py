import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from nutpred.preprocess import is_first_mapped
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

logger = logging.getLogger(__name__)


# Define target nutrients
TARGET_NUT3 = ['Calcium(mg)', 'Fiber(g)', 'Iron(mg)']
ORDER = ["nut8", "nut8+binary", "nut8+score", "nut8+umap_10", "ing_pred"]
GROUP_COLS = ["full", "first_mapped", "mapped_ratio_high", "mapped_ratio_top20_high", "strict"]
    

def compare_feature_sets(metrics_df: pd.DataFrame = None, metrics_file_path: str = None, outdir: str = None):
    """Create a single heatmap showing model types (rows) × metrics (columns)."""
    logger.info("Creating model type × metrics comparison heatmap")
    
    # Load metrics data
    if metrics_df is not None:
        df = metrics_df
    elif metrics_file_path is not None:
        df = pd.read_csv(metrics_file_path)
    else:
        # Default to the standard metrics file
        df = pd.read_csv("metrics_test.csv")
    
    if len(df) == 0:
        logger.warning("No metrics found for comparison")
        return
    
    # Create model type column that combines Model and FeatureSet
    df['ModelType'] = df.apply(lambda row: 
        f"{row['Model']}_{row['FeatureSet']}" if row['Model'] != 'ing_pred' 
        else 'ing_pred', axis=1)
    
    # Define metrics to include
    metrics_to_plot = ['R2', 'RMSE', 'SMAPE']
    
    # Create separate heatmaps for each nutrient: ModelType (rows) × Group (columns)
    for nutrient in TARGET_NUT3:
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics_to_plot):
            if metric not in df.columns:
                logger.warning(f"Metric {metric} not found in dataframe")
                continue
                
            # Create pivot table: ModelType (rows) × Group (columns)
            pivot = df.query(f"Nutrient == '{nutrient}'").pivot_table(
                values=metric, 
                index='ModelType', 
                columns='Group', 
                aggfunc='mean'
            )
            
            # Reorder rows to put optimization first, then feature sets
            model_order = [col for col in pivot.index]
            pivot = pivot.reindex(model_order)
            pivot = pivot[GROUP_COLS]
            
            # Create heatmap
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='Blues', 
                    cbar_kws={'label': metric}, ax=axes[i])
            axes[i].set_title(f'{metric} by Model Type and Group for {nutrient}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Group', fontsize=10)
            axes[i].set_ylabel('Model Type', fontsize=10)
            
            # Rotate x-axis labels for better readability
            axes[i].tick_params(axis='x', rotation=30)
            axes[i].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        output_path = f"{outdir}/model_comparison_{nutrient}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison heatmap saved to {output_path}")

def create_scatterplots(pred_file_path: str = None, outdir: str = None):
    """Create scatterplots comparing predicted vs actual values for all models."""
    logger.info("Creating scatterplots for all models....")

    plots_dir = os.path.join(outdir, "scatterplots")
    os.makedirs(plots_dir, exist_ok=True)

    # Get all prediction columns
    food_df = pd.read_csv(pred_file_path)
    pred_cols =  []
    for nutrient in TARGET_NUT3:
        pred_cols.extend([col for col in food_df.columns if f'{nutrient}_' in col])

    # Create scatterplots for each nutrient, model type, and group
    for nutrient in TARGET_NUT3:
        for pred_col in pred_cols:
            for group in GROUP_COLS:
                data = food_df[[nutrient, pred_col, group]]            
                model_type = pred_col.split('_')[1]
                if model_type == "xgb":
                   feature_set = '_'.join(pred_col.split('_')[2:])
                   plt.figure(figsize=(10, 8))
                   sns.scatterplot(x=data[nutrient], y=data[pred_col], alpha=0.5)
                   plt.title(f'Scatterplot: {nutrient} - {model_type} - {group} - {feature_set}')
                   plt.xlabel('True')
                   plt.ylabel('Predicted')
                   plt.axline((0, 0), (1, 1), color='black', linestyle='--')
                   plt.savefig(os.path.join(plots_dir, f'scatterplot_{nutrient}_{model_type}_{group}_{feature_set}.png'))
                elif model_type == "opt":
                    plt.figure(figsize=(10, 8))
                    sns.scatterplot(x=data[nutrient], y=data[pred_col], alpha=0.5)
                    plt.title(f'Scatterplot: {nutrient} - {model_type} - {group}')
                    plt.xlabel('True')
                    plt.ylabel('Predicted')
                    plt.axline((0, 0), (1, 1), color='black', linestyle='--')
                    plt.savefig(os.path.join(plots_dir, f'scatterplot_{nutrient}_{model_type}_{group}.png'))
                plt.close()
                logger.info(f"Scatterplot saved for {nutrient} - {model_type} - {group}")

def create_shap_plots(food_df: pd.DataFrame, models_dict: dict, feature_sets: dict, outdir: str):
    """Create SHAP plots for XGBoost models."""
    import shap
    logger.info("Creating SHAP plots for XGBoost models")
    
    # Create plots directory structure
    shap_dir = os.path.join(outdir, "shap")
    os.makedirs(shap_dir, exist_ok=True)
    
    # Get test data for SHAP analysis
    test_data = food_df[food_df['SampleType'] == 'test'].copy()
    
    if len(test_data) == 0:
        logger.warning("No test data available for SHAP analysis")
        return
    
    # Create SHAP plots for each model
    for nutrient in TARGET_NUT3:
        logger.info(f"Creating SHAP plots for {nutrient}")     
        for feature_set_name, feature_cols in feature_sets.items():
            for group_name, model_dict in models_dict.items():
                for model_name, model in model_dict.items():
                    model_key = f"{nutrient}_{feature_set_name}_{group_name}_{model_name}"
            
                    model = model[model_key]
            
                    # Prepare features for SHAP
                    X_test = test_data[feature_cols].values
                    
                    # Remove rows with NaN values
                    mask = ~np.isnan(X_test).any(axis=1)
                    X_test_clean = X_test[mask]
                    
                    if len(X_test_clean) == 0:
                        logger.warning(f"No valid test data for {model_key}")
                        continue
                    
                    # Create SHAP explainer
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test_clean)
                        
                        # Create SHAP summary plot
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_test_clean, feature_names=feature_cols, show=False)
                    plt.title(f'SHAP Summary Plot: {nutrient} - {feature_set_name} - {group_name}')
                    plt.tight_layout()
                    
                    summary_path = os.path.join(shap_dir, f"shap_summary_{nutrient}_{feature_set_name}_{group_name}.png")
                    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Create SHAP bar plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_test_clean, feature_names=feature_cols, plot_type="bar", show=False)
                    plt.title(f'SHAP Feature Importance: {nutrient} - {feature_set_name} - {group_name}')
                    plt.tight_layout()
                    
                    bar_path = os.path.join(shap_dir, f"shap_bar_{nutrient.replace('(', '').replace(')', '').replace(' ', '_')}_{feature_set_name}.png")
                    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"SHAP plots saved for {model_key}")
                
    
    logger.info(f"All SHAP plots saved to {shap_dir}")

__all__=[compare_feature_sets, create_scatterplots, create_shap_plots]