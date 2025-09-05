# nutpred — Ingredient-Nutrient Prediction Toolkit

Predict **Calcium (mg), Fiber (g), and Iron (mg)** from packaged food data using:
1) nutrient panels,  
2) ingredient lists (binary & positional scores),  
3) provided low-dim ingredient embeddings (`umap_10`), and  
4) an optimization approach that infers ingredient ratios from an ingredient–nutrient table.

This repo provides a clean, reproducible pipeline with concise, side-by-side truth/pred outputs and effective visualizations.

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/sarahjang15/nutpred.git

# Install dependencies
pip install -r nutpred/requirements.txt

### 2. Prepare Data

Place your data files in the `data/` directory:
- `snack_input_df.csv` - Main snack dataset
- `THESAURUSFORPUBLICRELEASE.XLSX` - Ingredient thesaurus
- `ingnut_df_top135.csv` - Nutrient information of ingredients (Snacks top 133 ingredients for testing)

### 3. Run the Pipeline

#### Test Mode (Quick Run)
```bash
# Run with 100 samples for testing
python run.py --test-size 100

# Run with custom parameters
python run.py --test-size 50 --filter-values popcorn pretzel
```

#### Complete Mode (Full Dataset)
```bash
# Run complete pipeline
python run.py

# Run with custom parameters
python run.py --filter-type ingredients --filter-values popcorn pretzel pretzels --cv 5
```

#### Advanced Options
```bash
# Full example with all options
python run.py \
  --test-size 200 \
  --random-state 42 \
  --filter-type ingredients \
  --filter-values popcorn pretzel pretzels \
  --food-df data/snack_input_df.csv \
  --thesaurus-df data/THESAURUSFORPUBLICRELEASE.XLSX \
  --ingnut-df data/ingnut_df_top135.csv \
  --outdir ./my_outputs \
  --cv 3 \
  --force-rf \
  --opt-constraint nnls_mono \
  --opt-solver osqp \
  --opt-ridge 0.1
```

---

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--test-size` | `None` | Number of samples for test mode (None = complete mode) |
| `--random-state` | `42` | Random seed for reproducibility |
| `--filter-type` | `ingredients` | Filter type: `ingredients` or `category` |
| `--filter-values` | `popcorn pretzel pretzels` | Values to filter by |
| `--food-df` | `data/snack_input_df.csv` | Path to main dataset |
| `--thesaurus-df` | `data/THESAURUSFORPUBLICRELEASE.XLSX` | Path to thesaurus |
| `--ingnut-df` | `data/ingnut_df_top135.csv` | Path to ingredient-nutrient table |
| `--outdir` | `./nutpred_outputs` | Output directory |
| `--cv` | `3` | Cross-validation folds |
| `--force-rf` | `False` | Use Random Forest instead of XGBoost |
| `--opt-constraint` | `nnls_mono` | Optimization constraint type |
| `--opt-solver` | `osqp` | Optimization solver  |
| `--opt-ridge` | `0.0` | Ridge regularization parameter |

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
├── run.py                    # Main pipeline runner
├── pyproject.toml
├── .gitignore
├── nutpred/                  # Python package
│   ├── __init__.py
│   ├── cleaning.py           # robust ingredient text cleaner
│   ├── metrics.py            # r2_manual (SSR/SST), SMAPE
│   ├── preprocess.py         # load → map to thesaurus → filter → top-K → binary/score → expand umap_10
│   ├── pred_by_ingnut.py     # CVXPY optimizer → ing_pred (and its metrics)
│   ├── pred_by_fullnut.py    # XGBoost/RF for: nut8, nut8+binary, nut8+score, nut8+umap_10
│   ├── viz.py                # heatmaps for R², RMSE, SMAPE
│   └── requirements.txt      # Python dependencies
├── data/                     # Input data (not tracked by git)
│   ├── snack_input_df.csv
│   ├── THESAURUSFORPUBLICRELEASE.XLSX
│   └── ingnut_df_top135.csv
├── outputs/                  # Pipeline outputs (not tracked by git)
│   ├── metrics_all_models.csv
│   ├── model_comparison_summary.csv
│   ├── ingredient_parsing_summary.csv
│   ├── ingredient_weights.csv
│   └── plots/
└── logs/                     # Log files (not tracked by git)
    └── nutpred.log
```

---

## Output Files

The pipeline generates several output files in the specified output directory:

### Core Outputs
- **`snack_df_complete.csv`** - Complete dataset with predictions
- **`metrics_all_models.csv`** - Performance metrics for all models
- **`model_comparison_summary.csv`** - Summary comparison of models
- **`ingredient_parsing_summary.csv`** - Ingredient parsing statistics
- **`ingredient_weights.csv`** - Optimized ingredient weights

### Visualizations
- **`feature_comparison.png`** - Feature set comparison heatmap
- **`scatterplot_*.png`** - Scatter plots for different feature sets
- **`filter_comparison_*.png`** - Filter comparison plots
- **`plots/shap_*.png`** - SHAP analysis plots (if generated)

### Logs
- **`nutpred.log`** - Detailed execution log

---

## Dependencies

Core dependencies (see `nutpred/requirements.txt`):
- `numpy>=1.23,<3`
- `pandas>=2.0,<3`
- `scikit-learn>=1.3,<2`
- `matplotlib>=3.7,<4`
- `seaborn>=0.12,<1`
- `cvxpy>=1.4,<2`
- `osqp>=0.6,<1`
- `openpyxl>=3.1,<4`
- `xgboost>=1.7,<2`
- `shap>=0.42,<1`
- `tqdm>=4.65,<5`

---

## Examples

### Basic Usage
```bash
# Quick test with 50 samples
python run.py --test-size 50

# Full pipeline with default settings
python run.py
```

### Custom Filtering
```bash
# Filter by specific ingredients
python run.py --filter-values chocolate vanilla

# Filter by category
python run.py --filter-type category --filter-values snacks beverages
```

### Advanced Configuration
```bash
# Use Random Forest instead of XGBoost
python run.py --force-rf

# Custom optimization settings
python run.py --opt-constraint eq1_mono --opt-solver clarabel --opt-ridge 0.1

# Custom output directory
python run.py --outdir ./my_results
```

---

## Troubleshooting

### Common Issues

1. **Missing data files**: Ensure all required files are in the `data/` directory
2. **Memory issues**: Use `--test-size` for smaller datasets during development
3. **Dependency conflicts**: Use a virtual environment and install from `requirements.txt`
4. **Permission errors**: Check write permissions for output directories

### Logs
Check `nutpred.log` for detailed execution information and error messages.
