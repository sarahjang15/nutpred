# yfpred — Yield Factor Prediction (with Korean Ingredients)

This module provides tools for predicting **yield factors** for Korean ingredients in processed food products using optimization-based and GPT-based approaches.

## Overview

**Yield Factor (YF)** represents how nutrients change during food processing:
- **YF < 1.0**: Nutrient loss during processing (cooking, evaporation, degradation)
- **YF = 1.0**: No nutrient change (already processed ingredients)
- **YF > 1.0**: Nutrient concentration (rare, constrained to ≤ 1.0)

The core equation solved is: **sum(ingredient_nutrition / yield_factor) ≈ food_nutrition**

---

## Usage

### Command Line Interface

**Basic Usage:**
```bash
# Optimization method with default parameters
python main.py

# Optimization method with custom parameters
python main.py --scale logiqr --ridge 0.01 --error l2 --solver osqp

# Specify custom input and output files
python main.py --food-df data/my_food.csv --ingnut-df data/my_ingnut.csv \
  --outdir ./results --output-excel results.xlsx --output-matrix matrix.npy

# GPT method
python main.py --method gpt --gpt-model gpt-5 --api-key YOUR_API_KEY

# Test with different nutrient sets
python main.py --nutrients nut6  # 6 nutrients
python main.py --nutrients nut8  # 8 nutrients (default)

# Run full test suite
python main.py --run-suite
```

**Arguments:**
- `--method`: "optimization" or "gpt" (default: optimization)
- `--scale`: Scaling method ("none", "std", "pminmax", "logiqr")
- `--ridge`: Ridge regularization parameter (default: 0.01)
- `--error`: Error type ("l1" or "l2")
- `--solver`: Optimization solver ("auto", "osqp", "ecos", "scs", "clarabel", "piqp")
- `--nutrients`: Nutrient set ("nut3", "nut4", "nut5", "nut6", "nut8", "energy_carb", "protein_fat")
- `--max-yield-factor`: Maximum allowed yield factor (default: 1.0)
- `--api-key`: OpenAI API key (for GPT method)
- `--gpt-model`: GPT model ("gpt-5", "gpt-4", "gpt-4o")
- `--test-case`: "real" or "perfect" (default: real)
- `--run-suite`: Run full test suite
- `--outdir`: Output directory for saving results (default: "./")
- `--food-df`: Path to food DataFrame CSV file (default: "potato_example_foodnut.csv")
- `--ingnut-df`: Path to ingredient-nutrient DataFrame CSV file (default: "potato_example_ingnut.csv")
- `--output-excel`: Name of output Excel file (default: "potato_example_output.xlsx")
- `--output-matrix`: Name of output numpy matrix file (default: "yield_factors_matrix.npy")

---

## Python API

### Optimization Method

```python
import pandas as pd
from pred_yf_kr import predict_yield_factors

# Load data
food_df = pd.read_csv('data/food_products.csv')
ingnut_df = pd.read_csv('data/korean_ingnut.csv')

# Predict yield factors
nut_cols = ["Energy(kcal)", "Carbohydrate(g)", "Total fat(g)", "Protein(g)", 
            "Sodium(mg)", "Total sugar(g)", "Saturated fatty acids(g)", "Cholesterol(mg)"]

food_df, yf_preds, failed = predict_yield_factors(
    food_df=food_df,
    korean_ingnut_df=ingnut_df,
    nut8_cols=nut_cols,
    korean_ingnut_cols=nut_cols,
    group_name="test",
    scale="std",
    ridge=0.01,
    solver_name="osqp",
    error_type="l2",
    max_yield_factor=1.0
)

print(f"Predictions complete. Failed: {len(failed)}")
print(f"Yield factor matrix shape: {yf_preds.shape}")
```

**Key Parameters:**
- `ridge`: Ridge regularization parameter (default: 0.0)
- `solver_name`: Optimization solver ("osqp", "ecos", "scs", "clarabel", "piqp", "auto")
- `scale`: Residual scaling method ("none", "std", "pminmax", "logiqr")
- `error_type`: Error type ("l1" for absolute error, "l2" for squared error)
- `max_yield_factor`: Maximum allowed yield factor (default: 1.0)

### GPT Method

```python
import pandas as pd
import os
from pred_yf_kr_gpt import predict_yield_factors_gpt

# Set API key
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# Load data
food_df = pd.read_csv('data/food_products.csv')
ingnut_df = pd.read_csv('data/korean_ingnut.csv')

# Predict yield factors using GPT
food_df, yf_preds, failed, evidence = predict_yield_factors_gpt(
    food_df=food_df,
    korean_ingnut_df=ingnut_df,
    nut8_cols=nut_cols,
    korean_ingnut_cols=nut_cols,
    group_name="gpt_test",
    model="gpt-5",
    max_yield_factor=1.0,
    outdir="./gpt_results"
)

# View evidence
print(evidence[0]['evidence']['analysis'])
```

**Key Parameters:**
- `api_key`: OpenAI API key (or set `OPENAI_API_KEY` environment variable)
- `model`: GPT model to use ("gpt-5", "gpt-4", "gpt-4o")
- `max_yield_factor`: Maximum allowed yield factor (default: 1.0)
- `outdir`: Output directory for saving evidence file (default: "./")

---

## Data Requirements

### Input Files

1. **Food DataFrame** with columns:
   - Nutrition columns: `Energy(kcal)`, `Carbohydrate(g)`, `Total fat(g)`, `Protein(g)`, `Sodium(mg)`, `Total sugar(g)`, `Saturated fatty acids(g)`, `Cholesterol(mg)`
   - `ing_list`: List of ingredient names for each product
   - `kor_name`: Korean product name (optional)

2. **Korean Ingredient-Nutrient DataFrame** with columns:
   - `ing`: Ingredient name (must match names in `ing_list`)
   - Nutrient columns: `Energy`, `Carbohydrate, by difference`, `Total lipid (fat)`, `Protein`, `Sodium, Na`, `Sugars, total`, `Fatty acids, total saturated`, `Cholesterol`
   - `yield_factor`: Known yield factors (optional, for evaluation)

### Output Files

- **Prediction results**: DataFrame with added yield factor prediction columns
- **Yield factor matrix**: NumPy array (N products × K ingredients)
- **Evidence file**: JSON file with GPT reasoning (GPT method only)
- **Output Excel**: Processed results with truth/prediction pairs

All output files are saved to the specified output directory (`--outdir`).

---

## Dependencies

- `numpy>=1.23`
- `pandas>=2.0`
- `cvxpy>=1.4`
- `osqp>=0.6`
- `scikit-learn>=1.3`
- `openai` (for GPT method)
- `tqdm>=4.65`
