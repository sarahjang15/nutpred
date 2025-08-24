import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

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

def compare_feature_sets(metrics_df: pd.DataFrame):
    """
    Create comparison visualizations for feature sets.
    Returns matplotlib figure object.
    """
    logger.info("Creating feature set comparison visualizations")
    logger.info(f"Input metrics shape: {metrics_df.shape}")
    
    # Filter for target nutrients
    target_nutrients = ["Calcium(mg)","Fiber(g)","Iron(mg)"]
    filtered_df = metrics_df[metrics_df["Nutrient"].isin(target_nutrients)]
    logger.info(f"Filtered to target nutrients: {filtered_df.shape}")
    
    # Create pivot tables
    r2_pivot = filtered_df.pivot(index="FeatureSet", columns="Nutrient", values="R2").reindex(ORDER)
    rmse_pivot = filtered_df.pivot(index="FeatureSet", columns="Nutrient", values="RMSE").reindex(ORDER)
    smape_pivot = filtered_df.pivot(index="FeatureSet", columns="Nutrient", values="SMAPE").reindex(ORDER)
    
    logger.debug(f"R² pivot shape: {r2_pivot.shape}")
    logger.debug(f"RMSE pivot shape: {rmse_pivot.shape}")
    logger.debug(f"SMAPE pivot shape: {smape_pivot.shape}")
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # R² heatmap
    logger.debug("Creating R² heatmap")
    im1 = axes[0].imshow(r2_pivot.values, aspect="auto", cmap='viridis')
    axes[0].set_xticks(range(len(r2_pivot.columns)))
    axes[0].set_xticklabels(r2_pivot.columns, rotation=45)
    axes[0].set_yticks(range(len(r2_pivot.index)))
    axes[0].set_yticklabels(r2_pivot.index)
    axes[0].set_title("R² (SSR/SST) by Feature Set")
    
    # Add text annotations for R²
    for i in range(r2_pivot.shape[0]):
        for j in range(r2_pivot.shape[1]):
            v = r2_pivot.values[i, j]
            if isinstance(v, (int, float)) and np.isfinite(v):
                axes[0].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10)
    
    # RMSE heatmap
    logger.debug("Creating RMSE heatmap")
    im2 = axes[1].imshow(rmse_pivot.values, aspect="auto", cmap='plasma')
    axes[1].set_xticks(range(len(rmse_pivot.columns)))
    axes[1].set_xticklabels(rmse_pivot.columns, rotation=45)
    axes[1].set_yticks(range(len(rmse_pivot.index)))
    axes[1].set_yticklabels(rmse_pivot.index)
    axes[1].set_title("RMSE by Feature Set")
    
    # Add text annotations for RMSE
    for i in range(rmse_pivot.shape[0]):
        for j in range(rmse_pivot.shape[1]):
            v = rmse_pivot.values[i, j]
            if isinstance(v, (int, float)) and np.isfinite(v):
                axes[1].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10)
    
    # SMAPE heatmap
    logger.debug("Creating SMAPE heatmap")
    im3 = axes[2].imshow(smape_pivot.values, aspect="auto", cmap='inferno')
    axes[2].set_xticks(range(len(smape_pivot.columns)))
    axes[2].set_xticklabels(smape_pivot.columns, rotation=45)
    axes[2].set_yticks(range(len(smape_pivot.index)))
    axes[2].set_yticklabels(smape_pivot.index)
    axes[2].set_title("SMAPE (%) by Feature Set")
    
    # Add text annotations for SMAPE
    for i in range(smape_pivot.shape[0]):
        for j in range(smape_pivot.shape[1]):
            v = smape_pivot.values[i, j]
            if isinstance(v, (int, float)) and np.isfinite(v):
                axes[2].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10)
    
    # Add colorbars
    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])
    fig.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    logger.info("Feature set comparison visualization created successfully")
    
    return fig
