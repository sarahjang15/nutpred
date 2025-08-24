import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from .metrics import r2_manual, smape

USING_XGB = True
try:
    from xgboost import XGBRegressor
except Exception:
    USING_XGB = False
    from sklearn.ensemble import RandomForestRegressor

def _get_estimator_and_grid(force_rf: bool=False):
    if USING_XGB and not force_rf:
        est = XGBRegressor(tree_method="hist", random_state=42)
        grid = {
            "n_estimators": [200, 500],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        model_name = "XGBoost"
    else:
        from sklearn.ensemble import RandomForestRegressor
        est = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid = {
            "n_estimators": [300, 600],
            "max_depth": [None, 10, 20],
            "max_features": ["sqrt", 0.5],
            "min_samples_leaf": [1, 2],
        }
        model_name = "RandomForest"
    return est, grid, model_name

def train_eval_sets(
    df: pd.DataFrame,
    nut8_cols: List[str],
    targets: List[str],
    binary_cols: List[str],
    score_cols: List[str],
    umap_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int = 3,
    force_rf: bool = False
) -> pd.DataFrame:
    """
    Evaluate *four* ML sets: 'nut8', 'nut8+binary', 'nut8+score', 'nut8+umap_10'.
    (The 'ing_pred' set is provided by the optimizer module.)
    """
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=test_size, random_state=random_state)
    est_base, param_grid, model_name = _get_estimator_and_grid(force_rf)

    results = []
    for nutrient in targets:
        y = df[nutrient].to_numpy(float)
        y_tr, y_te = y[tr], y[te]

        configs = []
        # 1) nut8
        Xtr = df.loc[tr, nut8_cols].to_numpy(float); Xte = df.loc[te, nut8_cols].to_numpy(float)
        configs.append(("nut8", Xtr, Xte, nut8_cols))

        # 2) nut8+binary
        if binary_cols:
            Xtr = np.hstack([df.loc[tr, nut8_cols].to_numpy(float), df.loc[tr, binary_cols].to_numpy(float)])
            Xte = np.hstack([df.loc[te, nut8_cols].to_numpy(float), df.loc[te, binary_cols].to_numpy(float)])
            configs.append(("nut8+binary", Xtr, Xte, nut8_cols+binary_cols))

        # 3) nut8+score
        if score_cols:
            Xtr = np.hstack([df.loc[tr, nut8_cols].to_numpy(float), df.loc[tr, score_cols].to_numpy(float)])
            Xte = np.hstack([df.loc[te, nut8_cols].to_numpy(float), df.loc[te, score_cols].to_numpy(float)])
            configs.append(("nut8+score", Xtr, Xte, nut8_cols+score_cols))

        # 4) nut8+umap_10
        if umap_cols:
            Xtr = np.hstack([df.loc[tr, nut8_cols].to_numpy(float), df.loc[tr, umap_cols].to_numpy(float)])
            Xte = np.hstack([df.loc[te, nut8_cols].to_numpy(float), df.loc[te, umap_cols].to_numpy(float)])
            configs.append(("nut8+umap_10", Xtr, Xte, nut8_cols+umap_cols))

        for name, Xtr, Xte, cols in configs:
            est = est_base.__class__(**est_base.get_params())
            grid = GridSearchCV(est, param_grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
            grid.fit(Xtr, y_tr)
            model = grid.best_estimator_
            preds = model.predict(Xte)

            r2 = r2_manual(y_te, preds)
            rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
            mae = float(mean_absolute_error(y_te, preds))
            mape = float(mean_absolute_percentage_error(y_te + 1e-6, preds))
            s = smape(y_te, preds)

            results.append({"Nutrient": nutrient, "FeatureSet": name, "Model": model.__class__.__name__,
                            "R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape, "SMAPE": s})
    return pd.DataFrame(results)
