import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from nutpred.preprocess import is_first_mapped
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

logger = logging.getLogger(__name__)

# Try to import SHAP, but don't fail if not available
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP library imported successfully")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP library not available. SHAP plots will be skipped.")

ORDER = ["nut8", "nut8+binary", "nut8+score", "nut8+umap_10", "ing_pred"]

def _plot_heatmap(matrix_df: pd.DataFrame, title: str, out_path: str):
    logger.debug(f"Creating heatmap: {title}")
    logger.debug(f"Matrix shape: {matrix_df.shape}")
    
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(matrix_df.values, aspect="auto")
    ax.set_xticks(range(len(matrix_df.columns))); ax.set_xticklabels(matrix_df.columns)
    ax.set_yticks(range(len(matrix_df.index))); ax.set_yticklabels(matrix_df.index)
    
    # Add text annotations
    for i in range(matrix_df.shape[0]):
        for j in range(matrix_df.shape[1]):
            v = matrix_df.values[i, j]
            if isinstance(v, (int, float)) and np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    
    logger.info(f"Saving heatmap to: {out_path}")
    plt.savefig(out_path, dpi=300); plt.close(fig)
    return out_path

def compare_feature_sets(metrics_df: pd.DataFrame = None, metrics_file_path: str = None):
    """Create heatmaps comparing different feature sets across nutrients."""
    logger.info("Creating feature comparison heatmaps")
    
    # Load metrics data
    if metrics_df is not None:
        df = metrics_df
    elif metrics_file_path is not None:
        df = pd.read_csv(metrics_file_path)
    else:
        # Default to the standard metrics file
        df = pd.read_csv("metrics_all_models.csv")
    
    # Filter to only include XGBoost models (exclude optimization)
    df = df[df['Model'] == 'XGBoost'].copy()
    
    if len(df) == 0:
        logger.warning("No XGBoost metrics found for feature comparison")
        return
    
    # Create pivot tables for each metric
    metrics_to_plot = ['R2', 'RMSE', 'MAE', 'SMAPE']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found in dataframe")
            continue
            
        # Create pivot table
        pivot = df.pivot_table(
            values=metric, 
            index='Nutrient', 
            columns='FeatureSet', 
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   center=pivot.values.mean(), ax=axes[i])
        axes[i].set_title(f'{metric} by Feature Set and Nutrient')
        axes[i].set_xlabel('Feature Set')
        axes[i].set_ylabel('Nutrient')
    
    plt.tight_layout()
    
    # Save plot
    output_path = "feature_comparison_test.png" if "test" in str(df) else "feature_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature comparison heatmap saved to {output_path}")

def _create_legacy_comparison(metrics_df: pd.DataFrame):
    """Create legacy comparison without Filter column"""
    logger.info("Creating legacy feature set comparison (no Filter column)")
    
    # Filter for target nutrients
    target_nutrients = ["Calcium(mg)","Fiber(g)","Iron(mg)"]
    filtered_df = metrics_df[metrics_df["Nutrient"].isin(target_nutrients)]
    
    # Create pivot tables
    r2_pivot = filtered_df.pivot(index="FeatureSet", columns="Nutrient", values="R2").reindex(ORDER)
    rmse_pivot = filtered_df.pivot(index="FeatureSet", columns="Nutrient", values="RMSE").reindex(ORDER)
    smape_pivot = filtered_df.pivot(index="FeatureSet", columns="Nutrient", values="SMAPE").reindex(ORDER)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # R² heatmap
    im1 = axes[0].imshow(r2_pivot.values, aspect="auto", cmap='Blues')
    axes[0].set_xticks(range(len(r2_pivot.columns)))
    axes[0].set_xticklabels(r2_pivot.columns, rotation=45)
    axes[0].set_yticks(range(len(r2_pivot.index)))
    axes[0].set_yticklabels(r2_pivot.index)
    axes[0].set_title("R² by Feature Set", fontsize=12, fontweight='bold')
    
    # Add text annotations for R²
    for i in range(r2_pivot.shape[0]):
        for j in range(r2_pivot.shape[1]):
            v = r2_pivot.values[i, j]
            if isinstance(v, (int, float)) and np.isfinite(v):
                axes[0].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, fontweight='bold')
    
    # RMSE heatmap
    im2 = axes[1].imshow(rmse_pivot.values, aspect="auto", cmap='Blues')
    axes[1].set_xticks(range(len(rmse_pivot.columns)))
    axes[1].set_xticklabels(rmse_pivot.columns, rotation=45)
    axes[1].set_yticks(range(len(rmse_pivot.index)))
    axes[1].set_yticklabels(rmse_pivot.index)
    axes[1].set_title("RMSE by Feature Set", fontsize=12, fontweight='bold')
    
    # Add text annotations for RMSE
    for i in range(rmse_pivot.shape[0]):
        for j in range(rmse_pivot.shape[1]):
            v = rmse_pivot.values[i, j]
            if isinstance(v, (int, float)) and np.isfinite(v):
                axes[1].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, fontweight='bold')
    
    # SMAPE heatmap
    im3 = axes[2].imshow(smape_pivot.values, aspect="auto", cmap='Blues')
    axes[2].set_xticks(range(len(smape_pivot.columns)))
    axes[2].set_xticklabels(smape_pivot.columns, rotation=45)
    axes[2].set_yticks(range(len(smape_pivot.index)))
    axes[2].set_yticklabels(smape_pivot.index)
    axes[2].set_title("SMAPE (%) by Feature Set", fontsize=12, fontweight='bold')
    
    # Add text annotations for SMAPE
    for i in range(smape_pivot.shape[0]):
        for j in range(smape_pivot.shape[1]):
            v = smape_pivot.values[i, j]
            if isinstance(v, (int, float)) and np.isfinite(v):
                axes[2].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, fontweight='bold')
    
    # Add colorbars
    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])
    fig.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    return fig

def _create_single_filter_comparison(filter_df: pd.DataFrame, filter_name: str):
    """Create comparison for a single filter"""
    logger.info(f"Creating single filter comparison for: {filter_name}")
    
    # Create pivot tables
    r2_pivot = filter_df.pivot(index="FeatureSet", columns="Nutrient", values="R2").reindex(ORDER)
    rmse_pivot = filter_df.pivot(index="FeatureSet", columns="Nutrient", values="RMSE").reindex(ORDER)
    smape_pivot = filter_df.pivot(index="FeatureSet", columns="Nutrient", values="SMAPE").reindex(ORDER)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Feature Set Comparison ({filter_name.upper()} Filter)', fontsize=14, fontweight='bold')
    
    # R² heatmap
    im1 = axes[0].imshow(r2_pivot.values, aspect="auto", cmap='Blues')
    axes[0].set_xticks(range(len(r2_pivot.columns)))
    axes[0].set_xticklabels(r2_pivot.columns, rotation=45)
    axes[0].set_yticks(range(len(r2_pivot.index)))
    axes[0].set_yticklabels(r2_pivot.index)
    axes[0].set_title(f"R² by Feature Set ({filter_name})", fontsize=12, fontweight='bold')
    
    # Add text annotations for R²
    for i in range(r2_pivot.shape[0]):
        for j in range(r2_pivot.shape[1]):
            v = r2_pivot.values[i, j]
            if isinstance(v, (int, float)) and np.isfinite(v):
                axes[0].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, fontweight='bold')
    
    # RMSE heatmap
    im2 = axes[1].imshow(rmse_pivot.values, aspect="auto", cmap='Blues')
    axes[1].set_xticks(range(len(rmse_pivot.columns)))
    axes[1].set_xticklabels(rmse_pivot.columns, rotation=45)
    axes[1].set_yticks(range(len(rmse_pivot.index)))
    axes[1].set_yticklabels(rmse_pivot.index)
    axes[1].set_title(f"RMSE by Feature Set ({filter_name})", fontsize=12, fontweight='bold')
    
    # Add text annotations for RMSE
    for i in range(rmse_pivot.shape[0]):
        for j in range(rmse_pivot.shape[1]):
            v = rmse_pivot.values[i, j]
            if isinstance(v, (int, float)) and np.isfinite(v):
                axes[1].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, fontweight='bold')
    
    # SMAPE heatmap
    im3 = axes[2].imshow(smape_pivot.values, aspect="auto", cmap='Blues')
    axes[2].set_xticks(range(len(smape_pivot.columns)))
    axes[2].set_xticklabels(smape_pivot.columns, rotation=45)
    axes[2].set_yticks(range(len(smape_pivot.index)))
    axes[2].set_yticklabels(smape_pivot.index)
    axes[2].set_title(f"SMAPE (%) by Feature Set ({filter_name})", fontsize=12, fontweight='bold')
    
    # Add text annotations for SMAPE
    for i in range(smape_pivot.shape[0]):
        for j in range(smape_pivot.shape[1]):
            v = smape_pivot.values[i, j]
            if isinstance(v, (int, float)) and np.isfinite(v):
                axes[2].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, fontweight='bold')
    
    # Add colorbars
    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])
    fig.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    return fig

def _create_multi_filter_comparison(filtered_df: pd.DataFrame, filters: list):
    """Create side-by-side comparison for multiple filters"""
    logger.info(f"Creating multi-filter comparison for: {filters}")
    
    # Create subplots: 3 metrics x 2 filters = 6 subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Feature Set Comparison by Filter', fontsize=16, fontweight='bold')
    
    # Ensure filters are in consistent order
    filter_order = ["strict", "all"] if "strict" in filters else filters
    
    for filter_idx, filter_name in enumerate(filter_order):
        if filter_name not in filters:
            continue
            
        filter_df = filtered_df[filtered_df["Filter"] == filter_name]
        
        # Create pivot tables for this filter
        r2_pivot = filter_df.pivot(index="FeatureSet", columns="Nutrient", values="R2").reindex(ORDER)
        rmse_pivot = filter_df.pivot(index="FeatureSet", columns="Nutrient", values="RMSE").reindex(ORDER)
        smape_pivot = filter_df.pivot(index="FeatureSet", columns="Nutrient", values="SMAPE").reindex(ORDER)
        
        # R² heatmap
        ax = axes[0, filter_idx]
        im1 = ax.imshow(r2_pivot.values, aspect="auto", cmap='Blues')
        ax.set_xticks(range(len(r2_pivot.columns)))
        ax.set_xticklabels(r2_pivot.columns, rotation=45)
        ax.set_yticks(range(len(r2_pivot.index)))
        ax.set_yticklabels(r2_pivot.index)
        ax.set_title(f"R² ({filter_name.upper()})", fontsize=12, fontweight='bold')
        
        # Add text annotations for R²
        for i in range(r2_pivot.shape[0]):
            for j in range(r2_pivot.shape[1]):
                v = r2_pivot.values[i, j]
                if isinstance(v, (int, float)) and np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, fontweight='bold')
        
        # RMSE heatmap
        ax = axes[1, filter_idx]
        im2 = ax.imshow(rmse_pivot.values, aspect="auto", cmap='Blues')
        ax.set_xticks(range(len(rmse_pivot.columns)))
        ax.set_xticklabels(rmse_pivot.columns, rotation=45)
        ax.set_yticks(range(len(rmse_pivot.index)))
        ax.set_yticklabels(rmse_pivot.index)
        ax.set_title(f"RMSE ({filter_name.upper()})", fontsize=12, fontweight='bold')
        
        # Add text annotations for RMSE
        for i in range(rmse_pivot.shape[0]):
            for j in range(rmse_pivot.shape[1]):
                v = rmse_pivot.values[i, j]
                if isinstance(v, (int, float)) and np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, fontweight='bold')
        
        # SMAPE heatmap
        ax = axes[2, filter_idx]
        im3 = ax.imshow(smape_pivot.values, aspect="auto", cmap='Blues')
        ax.set_xticks(range(len(smape_pivot.columns)))
        ax.set_xticklabels(smape_pivot.columns, rotation=45)
        ax.set_yticks(range(len(smape_pivot.index)))
        ax.set_yticklabels(smape_pivot.index)
        ax.set_title(f"SMAPE (%) ({filter_name.upper()})", fontsize=12, fontweight='bold')
        
        # Add text annotations for SMAPE
        for i in range(smape_pivot.shape[0]):
            for j in range(smape_pivot.shape[1]):
                v = smape_pivot.values[i, j]
                if isinstance(v, (int, float)) and np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, fontweight='bold')
        
        # Add colorbar for this filter
        fig.colorbar(im1, ax=axes[0, filter_idx])
        fig.colorbar(im2, ax=axes[1, filter_idx])
        fig.colorbar(im3, ax=axes[2, filter_idx])
    
    plt.tight_layout()
    return fig

def create_scatterplots(snack_df: pd.DataFrame, outdir: str, metrics_df: pd.DataFrame = None, metrics_file_path: str = None):
    """Create scatterplots comparing predicted vs actual values for all models."""
    logger.info("Creating scatterplots for all models")
    
    # Load metrics data if not provided
    if metrics_df is None:
        if metrics_file_path is not None:
            metrics_df = pd.read_csv(metrics_file_path)
        else:
            # Default to the standard metrics file
            metrics_df = pd.read_csv("metrics_all_models.csv")
    
    # Create plots directory
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Define model mappings
    model_mappings = {
        'ing_pred': {
            'Calcium(mg)': 'opt_Calcium(mg)',
            'Fiber(g)': 'opt_Fiber(g)', 
            'Iron(mg)': 'opt_Iron(mg)'
        }
    }
    
    # Add XGBoost models
    xgb_models = metrics_df[metrics_df['Model'] == 'XGBoost']['FeatureSet'].unique()
    for feature_set in xgb_models:
        model_mappings[f'xgb_{feature_set}'] = {}
        for nutrient in ['Calcium(mg)', 'Fiber(g)', 'Iron(mg)']:
            model_mappings[f'xgb_{feature_set}'][nutrient] = f'xgb_{feature_set}_{nutrient}'
    
    # Create scatterplots for each model and nutrient
    for model_name, nutrient_mapping in model_mappings.items():
        logger.info(f"Creating scatterplots for {model_name}")
        
        # Get metrics for this model
        if model_name == 'ing_pred':
            model_metrics = metrics_df[metrics_df['FeatureSet'] == 'ing_pred']
        else:
            feature_set = model_name.replace('xgb_', '')
            model_metrics = metrics_df[(metrics_df['Model'] == 'XGBoost') & (metrics_df['FeatureSet'] == feature_set)]
        
        # Create subplot for each nutrient
        nutrients = list(nutrient_mapping.keys())
        num_nutrients = len(nutrients)
        
        if num_nutrients <= 3:
            fig, axes = plt.subplots(1, num_nutrients, figsize=(5*num_nutrients, 5))
            if num_nutrients == 1:
                axes = [axes]
        else:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
        
        for i, (nutrient, pred_col) in enumerate(nutrient_mapping.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get actual and predicted values
            actual = snack_df[nutrient].values
            predicted = snack_df[pred_col].values if pred_col in snack_df.columns else np.full_like(actual, np.nan)
            
            # Filter out NaN values
            valid_mask = ~(pd.isna(actual) | pd.isna(predicted))
            actual_valid = actual[valid_mask]
            predicted_valid = predicted[valid_mask]
            
            if len(actual_valid) == 0:
                ax.text(0.5, 0.5, f'No valid data for {nutrient}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_name} - {nutrient}')
                continue
            
            # Create scatter plot
            if 'first_mapped' in snack_df.columns:
                # Get first_mapped values for valid samples
                first_mapped_valid = snack_df.loc[valid_mask, 'first_mapped']
                
                # Create separate scatter plots for each first_mapped value
                true_mask = first_mapped_valid == True
                false_mask = first_mapped_valid == False
                
                # Green circles for first_mapped=TRUE
                if np.any(true_mask):
                    ax.scatter(actual_valid[true_mask], predicted_valid[true_mask], 
                              c='green', marker='o', alpha=0.6, s=20, label='first_mapped=TRUE')
                
                # Red 'x' marks for first_mapped=FALSE  
                if np.any(false_mask):
                    ax.scatter(actual_valid[false_mask], predicted_valid[false_mask], 
                              c='red', marker='x', alpha=0.6, s=20, label='first_mapped=FALSE')
                
                ax.legend()
            else:
                # Fallback to original behavior
                ax.scatter(actual_valid, predicted_valid, alpha=0.6, s=20)
            
            # Add diagonal line
            min_val = min(actual_valid.min(), predicted_valid.min())
            max_val = max(actual_valid.max(), predicted_valid.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Add metrics text
            r2 = r2_score(actual_valid, predicted_valid)
            rmse = np.sqrt(mean_squared_error(actual_valid, predicted_valid))
            
            # Get metrics from the metrics dataframe if available
            if len(model_metrics) > 0:
                nutrient_metrics = model_metrics[model_metrics['Nutrient'] == nutrient]
                if len(nutrient_metrics) > 0:
                    r2 = nutrient_metrics['R2'].iloc[0]
                    rmse = nutrient_metrics['RMSE'].iloc[0]
            
            ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{model_name} - {nutrient}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(num_nutrients, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(plots_dir, f'scatterplot_{model_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Scatterplot saved to {plot_path}")
    
    logger.info("All scatterplots created successfully")

def create_filter_comparison_plots(snack_df: pd.DataFrame, outdir: str, metrics_df: pd.DataFrame = None, metrics_file_path: str = None):
    """Create comparison plots showing performance differences between filters."""
    logger.info("Creating filter comparison plots")
    
    # Load metrics data if not provided
    if metrics_df is None:
        if metrics_file_path is not None:
            metrics_df = pd.read_csv(metrics_file_path)
        else:
            # Default to the standard metrics file
            metrics_df = pd.read_csv("metrics_all_models.csv")
    
    # Create plots directory
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Filter to only include models that have both "strict" and "all" filters
    if "Filter" not in metrics_df.columns:
        logger.warning("No 'Filter' column found in metrics, skipping filter comparison plots")
        return
    
    # Get models that have both filters
    models_with_both_filters = []
    for model in metrics_df['FeatureSet'].unique():
        model_data = metrics_df[metrics_df['FeatureSet'] == model]
        filters = model_data['Filter'].unique()
        if len(filters) >= 2:
            models_with_both_filters.append(model)
    
    if not models_with_both_filters:
        logger.warning("No models found with multiple filters for comparison")
        return
    
    # Create comparison plots for each metric
    metrics_to_plot = ['R2', 'RMSE', 'MAE', 'SMAPE']
    
    for metric in metrics_to_plot:
        if metric not in metrics_df.columns:
            continue
            
        fig, axes = plt.subplots(1, len(models_with_both_filters), figsize=(5*len(models_with_both_filters), 5))
        if len(models_with_both_filters) == 1:
            axes = [axes]
        
        for i, model in enumerate(models_with_both_filters):
            ax = axes[i]
            
            model_data = metrics_df[metrics_df['FeatureSet'] == model]
            
            # Create bar plot comparing filters
            filters = model_data['Filter'].unique()
            nutrients = model_data['Nutrient'].unique()
            
            x = np.arange(len(nutrients))
            width = 0.35
            
            for j, filter_name in enumerate(filters):
                filter_data = model_data[model_data['Filter'] == filter_name]
                values = [filter_data[filter_data['Nutrient'] == nutrient][metric].iloc[0] 
                         if len(filter_data[filter_data['Nutrient'] == nutrient]) > 0 else 0 
                         for nutrient in nutrients]
                
                ax.bar(x + j*width, values, width, label=f'{filter_name}', alpha=0.8)
            
            ax.set_xlabel('Nutrient')
            ax.set_ylabel(metric)
            ax.set_title(f'{model} - {metric}')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(nutrients, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(plots_dir, f'filter_comparison_{metric}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Filter comparison plot for {metric} saved to {plot_path}")
    
    logger.info("All filter comparison plots created successfully")

def create_shap_plots(snack_df: pd.DataFrame, models_dict: dict, feature_sets: dict, outdir: str):
    """
    Create SHAP plots for XGBoost models.
    
    Args:
        snack_df: DataFrame with features and predictions
        models_dict: Dictionary containing trained models {feature_set: {nutrient: model}}
        feature_sets: Dictionary of feature sets and their column names
        outdir: Output directory for plots
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, skipping SHAP plots")
        return
    
    logger.info("Creating SHAP plots for XGBoost models")
    
    # Create plots directory
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Target nutrients for SHAP analysis
    target_nutrients = ["Calcium(mg)", "Fiber(g)", "Iron(mg)"]
    
    for fname, feature_cols in feature_sets.items():
        if fname not in models_dict:
            logger.warning(f"No models found for feature set: {fname}")
            continue
            
        logger.info(f"Creating SHAP plots for feature set: {fname}")
        
        for nutrient in target_nutrients:
            if nutrient not in models_dict[fname]:
                logger.warning(f"No model found for {fname} - {nutrient}")
                continue
                
            model = models_dict[fname][nutrient]
            logger.info(f"Creating SHAP plot for {fname} - {nutrient}")
            
            try:
                # Prepare feature data
                X_df = snack_df[feature_cols].copy()
                
                # Remove any rows with NaN values
                valid_mask = ~X_df.isna().any(axis=1)
                X_df_clean = X_df[valid_mask]
                
                if len(X_df_clean) == 0:
                    logger.warning(f"No valid data for SHAP analysis: {fname} - {nutrient}")
                    continue
                
                # Use subset for SHAP analysis (first 200 samples or all if less)
                subset_size = min(200, len(X_df_clean))
                subset = X_df_clean.iloc[:subset_size]
                
                logger.info(f"Using {len(subset)} samples for SHAP analysis")
                
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(subset)
                
                # Create SHAP beeswarm plot
                plt.figure(figsize=(8, 6))
                shap.summary_plot(
                    shap_vals,
                    subset,
                    plot_type='dot',
                    show=False
                )
                plt.title(f"SHAP Beeswarm: XGBoost | {fname} | {nutrient}")
                plt.tight_layout()
                
                # Save plot
                plot_filename = f"shap_beeswarm_XGB_{fname}_{nutrient}.png"
                plot_path = os.path.join(plots_dir, plot_filename)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved SHAP plot: {plot_path}")
                
                # Also create SHAP bar plot
                plt.figure(figsize=(8, 6))
                shap.summary_plot(
                    shap_vals,
                    subset,
                    plot_type='bar',
                    show=False
                )
                plt.title(f"SHAP Feature Importance: XGBoost | {fname} | {nutrient}")
                plt.tight_layout()
                
                # Save bar plot
                bar_plot_filename = f"shap_bar_XGB_{fname}_{nutrient}.png"
                bar_plot_path = os.path.join(plots_dir, bar_plot_filename)
                plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved SHAP bar plot: {bar_plot_path}")
                
            except Exception as e:
                logger.error(f"Error creating SHAP plot for {fname} - {nutrient}: {e}")
                continue
    
    logger.info("SHAP plots creation completed")


__all__=[compare_feature_sets, create_scatterplots, create_shap_plots, create_filter_comparison_plots]