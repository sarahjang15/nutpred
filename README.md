# nutpred — Ingredient-Nutrient Prediction Toolkit

Predict **Calcium (mg), Fiber (g), and Iron (mg)** from packaged food data using:
1) nutrient panels,  
2) ingredient lists (binary & positional scores),  
3) provided low-dim ingredient embeddings (`umap_10`), and  
4) an optimization approach that infers ingredient ratios from an ingredient–nutrient table.

This repo provides a clean, reproducible pipeline with concise, side-by-side truth/pred outputs and effective visualizations.

---

## Getting Started

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sarahjang15/nutpred.git
   cd nutpred
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r nutpred/requirements.txt
   ```

3. **Prepare Data**:
   Place your data files in the `data/` directory:
   - `snack_input_df.csv` - Main snack dataset
   - `THESAURUSFORPUBLICRELEASE.XLSX` - Ingredient thesaurus
   - `ingnut_df_top135.csv` - Nutrient information of ingredients

### Running the Pipeline

1. **Test Mode (Quick Run)**
  ```bash
  # Run with 100 samples for testing (>100 recommended)
  python run.py --test-size 100

  # Run with custom parameters
  python run.py --test-size 100 --filter-values popcorn pretzel
  ```

2. **Complete Mode (Full Dataset)**
  ```bash
  # Run complete pipeline
  python run.py

  # Run with custom parameters
  python run.py --filter-type ingredients --filter-values popcorn pretzel pretzels --cv 5
  ```

3. **Advanced Options**
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
| `--filter-values` | `popcorn pretzel pretzels` | Values to filter by:<br/>• **`ingredients`**: Deletes rows containing these ingredient names<br/>• **`category`**: Keeps only rows with these category names |
| `--food-df` | `data/snack_input_df.csv` | Path to main dataset |
| `--thesaurus-df` | `data/THESAURUSFORPUBLICRELEASE.XLSX` | Path to thesaurus |
| `--ingnut-df` | `data/ingnut_df_top135.csv` | Path to ingredient-nutrient table |
| `--outdir` | `./nutpred_outputs` | Output directory |
| `--cv` | `3` | Cross-validation folds |
| `--force-rf` | `False` | Use Random Forest instead of XGBoost |
| `--opt-constraint` | `nnls_mono` | Optimization constraint type:<br/>• `nnls_only`: Non-negative least squares only<br/>• `nnls_mono`: Non-negative least squares with monotonicity<br/>• `eq1`: Equality constraint (sum = 1)<br/>• `eq1_mono`: Equality constraint with monotonicity<br/>• `le1`: Less than or equal constraint (sum ≤ 1)<br/>• `le1_mono`: Less than or equal constraint with monotonicity |
| `--opt-solver` | `osqp` | Optimization solver:<br/>• `osqp`: Operator Splitting Quadratic Program (fast, reliable)<br/>• `clarabel`: Interior-point solver (high precision)<br/>• `scs`: Splitting Conic Solver (general purpose)<br/>• `ecos`: Embedded Conic Solver (lightweight)<br/>• `piqp`: Proximal Interior Point Quadratic Programming |
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
- **Clear, diverse metrics to compare models**
  - **R² **, **RMSE**, **MAE**, **MAPE**, **SMAPE**.
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
├── test_outputs/             # Test mode outputs 
│
├── complete_outputs/         # Complete mode outputs 
│
└── logs/                     # Log files (not tracked by git)
    └── nutpred.log
```

---

## Output Files

The pipeline generates output files in dynamically created directories based on your filter settings:

### Directory Structure
- **Test Mode**: `test_outputs/[filter_type]_[filter_values]/`
- **Complete Mode**: `complete_outputs/[filter_type]_[filter_values]/`
- **Full Dataset**: `test_outputs/` or `complete_outputs/` (when `--filter-values full`)

### Core Outputs
- **`config.yaml`** - Complete configuration parameters for this run
- **`df_with_preds_[mode].csv`** - Dataset with truth/prediction pairs
- **`metrics_[mode].csv`** - Performance metrics for all models
- **`model_comparison_[mode].csv`** - Summary comparison of models
- **`ingredient_parsing_[mode].csv`** - Ingredient parsing statistics
- **`ingredient_weights_[mode].csv`** - Optimized ingredient weights

### Visualizations
- **`feature_comparison.png`** - Feature set comparison heatmap
- **`scatterplot_*.png`** - Scatter plots for different feature sets
- **`filter_comparison_*.png`** - Filter comparison plots
- **`plots/shap_*.png`** - SHAP analysis plots (if generated)

### Logs
- **`logs/nutpred_[timestamp].log`** - Detailed execution log with timestamp

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
# Filter by specific ingredients (removes rows containing these ingredients)
python run.py --filter-values chocolate vanilla

# Filter by category (keeps only rows with these categories)
python run.py --filter-type category --filter-values snacks beverages
```

**Filtering Behavior:**
- **`--filter-type ingredients`**: **Excludes** rows that contain any of the specified ingredient names
- **`--filter-type category`**: **Includes only** rows that match the specified category names

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
