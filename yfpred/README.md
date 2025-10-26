# nutpred_kr — Korean Ingredient Yield Factor Prediction

This module provides tools for predicting **yield factors** for Korean ingredients in processed food products using optimization-based and GPT-based approaches.

## Overview

**Yield Factor (YF)** represents how nutrients change during food processing:
- **YF < 1.0**: Nutrient loss during processing (cooking, evaporation, degradation)
- **YF = 1.0**: No nutrient change (already processed ingredients)
- **YF > 1.0**: Nutrient concentration (rare, constrained to ≤ 1.0)

The core equation solved is: **sum(ingredient_nutrition / yield_factor) ≈ food_nutrition**

## Files

### `pred_yf_kr.py`
Optimization-based yield factor prediction using CVXPY.

**Key Features:**
- Solves for ingredient yield factors using convex optimization
- Supports multiple solvers: OSQP, ECOS, SCS, Clarabel, PIQP
- Flexible error types: L1 (absolute) or L2 (squared) loss
- Scaling methods: none, std, pminmax, logiqr
- Robust optimization with Huber loss support
- Ridge regularization

**Main Function:**
```python
from pred_yf_kr import predict_yield_factors

food_df, yf_preds, failed = predict_yield_factors(
    food_df=food_df,
    korean_ingnut_df=korean_ingnut_df,
    nut8_cols=["Energy(kcal)", "Carbohydrate(g)", ...],
    korean_ingnut_cols=["Energy", "Carbohydrate, by difference", ...],
    resolver="rule",
    ridge=0.01,
    robust=False,
    solver_name="osqp",
    scale="std",
    group_name="korean_yf",
    error_type="l2",
    max_yield_factor=1.0
)
```

**Parameters:**
- `ridge`: Ridge regularization parameter (default: 0.0)
- `robust`: Use Huber loss for robustness (default: False)
- `solver_name`: Optimization solver ("osqp", "ecos", "scs", "clarabel", "piqp", "auto")
- `scale`: Residual scaling method ("none", "std", "pminmax", "logiqr")
- `error_type`: Error type ("l1" for absolute error, "l2" for squared error)
- `max_yield_factor`: Maximum allowed yield factor (default: 1.0)

---

### `pred_yf_kr_gpt.py`
GPT-based yield factor prediction using OpenAI API.

**Key Features:**
- Uses GPT-5 (or GPT-4) for food science reasoning
- Provides detailed evidence and reasoning for each prediction
- Considers ingredient processing states (raw vs. processed)
- Incorporates industry knowledge and recipes
- Automatically enforces YF ≤ 1.0 constraint

**Main Function:**
```python
from pred_yf_kr_gpt import predict_yield_factors_gpt

food_df, yf_preds, failed, evidence_list = predict_yield_factors_gpt(
    food_df=food_df,
    korean_ingnut_df=korean_ingnut_df,
    nut8_cols=["Energy(kcal)", "Carbohydrate(g)", ...],
    korean_ingnut_cols=["Energy", "Carbohydrate, by difference", ...],
    api_key=api_key,
    model="gpt-5",
    group_name="korean_yf",
    max_yield_factor=1.0
)
```

**Parameters:**
- `api_key`: OpenAI API key (or set `OPENAI_API_KEY` environment variable)
- `model`: GPT model to use ("gpt-5", "gpt-4", "gpt-4o")
- `max_yield_factor`: Maximum allowed yield factor (default: 1.0)

**Outputs:**
- Adds `opt_pred_yield_factors_{group_name}` column with per-food yield factors
- Adds `gpt_evidence_{group_name}` column with detailed reasoning
- Saves evidence to `gpt_evidence_{group_name}.json`

---

### `try_yf_kr.py`
Test script for verifying yield factor predictions.

**Usage:**
```bash
# Optimization method with default parameters
python try_yf_kr.py

# Optimization method with custom parameters
python try_yf_kr.py --scale logiqr --ridge 0.01 --error l2 --solver osqp

# GPT method
python try_yf_kr.py --method gpt --gpt-model gpt-5 --api-key YOUR_API_KEY

# Test with different nutrient sets
python try_yf_kr.py --nutrients nut6  # 6 nutrients
python try_yf_kr.py --nutrients nut5  # 5 nutrients
python try_yf_kr.py --nutrients nut4  # 4 nutrients
python try_yf_kr.py --nutrients nut3  # 3 nutrients

# Run full test suite
python try_yf_kr.py --run-suite
```

**Arguments:**
- `--method`: "optimization" or "gpt" (default: optimization)
- `--scale`: Scaling method ("none", "std", "pminmax", "logiqr")
- `--ridge`: Ridge regularization parameter
- `--error`: Error type ("l1" or "l2")
- `--solver`: Optimization solver ("auto", "osqp", "ecos", "scs", "clarabel", "piqp")
- `--nutrients`: Nutrient set ("nut3", "nut4", "nut5", "nut6", "nut8", "energy_carb", "protein_fat")
- `--max-yield-factor`: Maximum allowed yield factor
- `--api-key`: OpenAI API key (for GPT method)
- `--gpt-model`: GPT model ("gpt-5", "gpt-4", "gpt-4o")
- `--test-case`: "real" or "perfect" (default: real)
- `--run-suite`: Run full test suite

**Test Verification:**
The script verifies that the equation `sum(ing_nut / yf) ≈ food_nut` holds for each product and nutrient, calculating errors and reporting overall statistics.

---

## Data Requirements

### Input Files
1. **Food DataFrame** (`food_df`) with columns:
   - Nutrition columns: `Energy(kcal)`, `Carbohydrate(g)`, `Total fat(g)`, `Protein(g)`, `Sodium(mg)`, `Total sugar(g)`, `Saturated fatty acids(g)`, `Cholesterol(mg)`
   - `ing_list`: List of ingredient names for each product
   - `kor_name`: Korean product name (optional)

2. **Korean Ingredient-Nutrient DataFrame** (`korean_ingnut_df`) with columns:
   - `ing`: Ingredient name (must match names in `ing_list`)
   - Nutrient columns: `Energy`, `Carbohydrate, by difference`, `Total lipid (fat)`, `Protein`, `Sodium, Na`, `Sugars, total`, `Fatty acids, total saturated`, `Cholesterol`
   - `yield_factor`: Known yield factors (optional, for evaluation)

### Output Files
- **Prediction results**: DataFrame with added yield factor prediction columns
- **Yield factor matrix**: NumPy array (N products × K ingredients)
- **Evidence file**: JSON file with GPT reasoning (GPT method only)
- **Output Excel**: Processed results with truth/prediction pairs

---

## Examples

### Example 1: Optimization Method
```python
import pandas as pd
from pred_yf_kr import predict_yield_factors

# Load data
food_df = pd.read_csv('data/food_products.csv')
ingnut_df = pd.read_csv('data/korean_ingnut.csv')

# Predict yield factors
nut_cols = ["Energy(kcal)", "Carbohydrate(g)", "Total fat(g)", "Protein(g)", 
            "Sodium(mg)", "Total sugar(g)", "Saturated fatty acids(g)", "Cholesterol(mg)"]

food_df_with_predictions, yf_preds, failed = predict_yield_factors(
    food_df=food_df,
    korean_ingnut_df=ingnut_df,
    nut8_cols=nut_cols,
    korean_ingnut_cols=nut_cols,
    group_name="test",
    scale="std",
    ridge=0.01,
    solver_name="osqp"
)

print(f"Predictions complete. Failed: {len(failed)}")
print(f"Yield factor matrix shape: {yf_preds.shape}")
```

### Example 2: GPT Method
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
food_df_with_predictions, yf_preds, failed, evidence = predict_yield_factors_gpt(
    food_df=food_df,
    korean_ingnut_df=ingnut_df,
    nut8_cols=nut_cols,
    korean_ingnut_cols=nut_cols,
    group_name="gpt_test",
    model="gpt-5"
)

# View evidence
print(evidence[0]['evidence']['analysis'])
```

---

## Key Concepts

### Ingredient State Interpretation
- **No state descriptors** (e.g., "감자", "CORN") → **RAW/FRESH state** → YF < 1.0
- **Has state descriptors** (e.g., "옥수수전분", "WHEAT FLOUR") → **PROCESSED state** → YF ≈ 1.0

### Processing Effects
- **Deep frying**: Oil absorption increases fat, but YF ≤ 1.0
- **Baking/cooking**: Water loss, nutrient degradation (YF < 1.0)
- **Drying**: Concentration of nutrients (YF ≤ 1.0)
- **Raw ingredients**: Significant loss during processing (YF < 1.0)
- **Processed ingredients**: Minimal change (YF ≈ 1.0)

### Constraint Enforcement
- **Critical**: Yield factors MUST satisfy **0 < YF ≤ 1.0**
- **Optimization**: Enforces `b >= 1/max_yield_factor` constraint
- **GPT**: Automatically caps predictions at 1.0

---

## Dependencies

- `numpy>=1.23`
- `pandas>=2.0`
- `cvxpy>=1.4`
- `osqp>=0.6`
- `scikit-learn>=1.3`
- `openai` (for GPT method)
- `tqdm>=4.65`

---

## See Also

- Main nutpred documentation: [`../README.md`](../README.md)
- Optimization module: [`pred_yf_kr.py`](pred_yf_kr.py)
- GPT module: [`pred_yf_kr_gpt.py`](pred_yf_kr_gpt.py)
- Test script: [`try_yf_kr.py`](try_yf_kr.py)

