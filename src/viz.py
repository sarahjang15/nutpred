import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ORDER = ["nut8", "nut8+binary", "nut8+score", "nut8+umap_10", "ing_pred"]

def _plot_heatmap(matrix_df: pd.DataFrame, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(matrix_df.values, aspect="auto")
    ax.set_xticks(range(len(matrix_df.columns))); ax.set_xticklabels(matrix_df.columns)
    ax.set_yticks(range(len(matrix_df.index))); ax.set_yticklabels(matrix_df.index)
    for i in range(matrix_df.shape[0]):
        for j in range(matrix_df.shape[1]):
            v = matrix_df.values[i, j]
            if isinstance(v, (int, float)) and np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close(fig)
    return out_path

def compare_feature_sets(metrics_df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    metrics_df = metrics_df[metrics_df["Nutrient"].isin(["Calcium(mg)","Fiber(g)","Iron(mg)"])]

    r2_pivot   = metrics_df.pivot(index="FeatureSet", columns="Nutrient", values="R2").reindex(ORDER)
    rmse_pivot = metrics_df.pivot(index="FeatureSet", columns="Nutrient", values="RMSE").reindex(ORDER)
    smape_pivot= metrics_df.pivot(index="FeatureSet", columns="Nutrient", values="SMAPE").reindex(ORDER)

    r2_png   = _plot_heatmap(r2_pivot, "R^2 (SSR/SST) by Feature Set", os.path.join(outdir, "heatmap_R2.png"))
    rmse_png = _plot_heatmap(rmse_pivot, "RMSE by Feature Set", os.path.join(outdir, "heatmap_RMSE.png"))
    smape_png= _plot_heatmap(smape_pivot, "SMAPE (%) by Feature Set", os.path.join(outdir, "heatmap_SMAPE.png"))
    return r2_png, rmse_png, smape_png
