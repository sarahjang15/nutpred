# nutpred — Ingredient-Nutrient Prediction Toolkit

Predict **Calcium (mg), Fiber (g), and Iron (mg)** from packaged snack data using:
1) nutrient panels,  
2) ingredient lists (binary & positional scores),  
3) provided low-dim ingredient embeddings (`umap_10`), and  
4) an optimization approach that infers ingredient ratios from an ingredient–nutrient table.

This repo provides a clean, reproducible pipeline with manual metrics (including **R² = SSR/SST**) and side-by-side truth/pred outputs.

---

## Features

- **Two complementary predictors**
  - **`ing_pred`**: CVXPY optimization over reference ingredient nutrients to infer ingredient ratios → predicts Ca/Fiber/Iron.
  - **Full-nutrient ML**: XGBoost (or RF fallback) on engineered features.
- **Comparable ML feature sets**
  - `nut8`  
  - `nut8+binary`  
  - `nut8+score`  
  - `nut8+umap_10`  
  - plus **`ing_pred`** (from the optimizer, evaluated side-by-side).
- **Robust preprocessing**
  - Ingredient string parsing + thesaurus mapping to normalized tokens.
  - Row filtering (e.g., popcorn/pretzel) to prevent leakage.
  - Top-K ingredient universe → binary & positional score features.
  - Uses **provided** `umap_10`; no embedding training inside the pipeline.
- **Clear, trustworthy metrics**
  - **R² (SSR/SST)**, **RMSE**, **MAE**, **MAPE**, **SMAPE**.
- **Tidy outputs**
  - Metrics CSV + heatmaps (R² / RMSE / SMAPE).
  - Processed dataset with **truth/pred pairs side-by-side**.
  - Saved ingredient-ratio weights from the optimizer.

---

## Repository Structure

```
nutpred/
├── README.md
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── .gitignore
├── nutpred/ # Python package
│ ├── init.py
│ ├── cleaning.py # robust ingredient text cleaner
│ ├── metrics.py # r2_manual (SSR/SST), SMAPE
│ ├── preprocess.py # load → map to thesaurus → filter → top-K → binary/score → expand umap_10
│ ├── pred_by_ingnut.py # CVXPY optimizer → ing_pred (and its metrics)
│ ├── pred_by_fullnut.py # XGBoost/RF for: nut8, nut8+binary, nut8+score, nut8+umap_10
│ ├── viz.py # heatmaps for R², RMSE, SMAPE
│ └── cli.py # end-to-end command-line entrypoint
├── data/ # (not tracked) put your inputs here
│ ├── snack_input_df.csv
│ ├── THESAURUSFORPUBLICRELEASE.xlsx
│ └── ingnut_df_top135.csv
├── outputs/ # pipeline writes here (metrics, plots, processed CSV)
│ └── .gitkeep
├── notebooks/ # optional exploratory work
│ └── exploration.ipynb
└── tests/ # optional unit tests
└── test_metrics.py
```