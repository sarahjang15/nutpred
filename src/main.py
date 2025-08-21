# src/main.py

import os
import argparse
import sys
import ast
import re
import numpy as np
import pandas as pd
import cvxpy as cp
import logging
import itertools
from cvxpy.error import SolverError
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from numpy.linalg import norm

# ─── Argument Parsing ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingredient–nutrient deconvolution with OSQP hyperparameter tuning"
    )
    parser.add_argument("--workdir", "-C", type=str, default=None,
                        help="Working directory override")
    parser.add_argument("--snack-data", "-S", type=str, required=True,
                        help="Path to snack data CSV")
    parser.add_argument("--ing-nut-data", "-I", type=str, required=True,
                        help="Path to ingredient–nutrient reference CSV")
    parser.add_argument("--output-prefix", "-O", type=str, required=True,
                        help="Prefix for output files")
    parser.add_argument("--top-n", type=int, required=True,
                        help="Number of top ingredients to use")
    parser.add_argument("--n-sample", type=int, default=None,
                        help="Subsample size for snacks")
    parser.add_argument("--resolve-mode", type=str, default="rule",
                        choices=["rule", "fit", "cosine"],
                        help="How to resolve ambiguous ingredient variants")
    parser.add_argument("--constraint", type=str, default="nnls_only",
                        choices=["nnls_only", "nnls_mono", "eq1", "eq1_mono", "le1", "le1_mono"],
                        help="Constraint: nnls_only (w>=0) or add mono (b[k] <= b[k+1]) eq1 (=1), eq1_mono (=1+mono), le1 (<=1), le1_mono (<=1+mono)")
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter tuning")
    parser.add_argument("--scale", type=str, default="logiqr", choices=["none", "std", "mad", "range", "minmax", "pminmax", "iqr", "logstd", "logiqr"],
                        help="Residual weighting. 'logiqr' is recommended for skewed data.")
    parser.add_argument("--robust", action="store_true",
                    help="Use Huber loss on residuals for robustness to outliers")
    parser.add_argument("--solver", type=str, default="osqp",
                        choices=["osqp", "clarabel", "scs", "ecos", "piqp"],
                        help="QP solver to use")

    return parser.parse_args()

args = parse_args()

# ─── Working Directory ──────────────────────────────────────────────────────────
if args.workdir:
    os.chdir(os.path.expanduser(args.workdir))
else:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("[INFO] Working dir:", os.getcwd())

# ─── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# ---- Solver options (high-precision defaults) ---------------------------------
solver_options = {
    "osqp":     {"eps_abs": 1e-6, "eps_rel": 1e-6, "max_iter": 500_000, "polish": True, "verbose": False},
    "clarabel": {"verbose": False, "tol_gap_abs": 1e-7, "tol_gap_rel": 1e-7},
    "scs":      {"eps": 1e-6, "verbose": False},
    "ecos":     {"abstol": 1e-7, "reltol": 1e-7, "verbose": False},
    "piqp":     {"eps_abs": 1e-7, "eps_rel": 1e-7, "verbose": False},
}

SOLVER_MAP = {
    "osqp":     cp.OSQP,
    "clarabel": cp.CLARABEL,
    "scs":      cp.SCS,
    "ecos":     cp.ECOS,
    "piqp":     cp.PIQP,
}
solver_key   = args.solver.lower()
solver_enum  = SOLVER_MAP[solver_key]
solver_kwargs = solver_options.get(solver_key, {})

# ─── 0) Load data ───────────────────────────────────────────────────────────────
snack = pd.read_csv(args.snack_data)
if args.n_sample:
    snack = snack.sample(n=args.n_sample, random_state=2025).reset_index(drop=True)
ingnut_df = pd.read_csv(args.ing_nut_data)

# parse ingredient lists
snack['mapped_list_top20'] = snack['mapped_list_top20'].apply(ast.literal_eval)

# ─── Nutrient column sets ────────────────────────────────────────────────────
nut_cols = ['Energy(kcal)', 'Total fat(g)', 'Protein(g)',
            'Carbohydrate(g)', 'Total sugar(g)', 'Sodium(mg)',
            'Cholesterol(mg)', 'Saturated fatty acids(g)']

ingnut_cols = ['Energy', 'Total lipid (fat)', 'Protein',
               'Carbohydrate, by difference', 'Sugars, total', 'Sodium, Na',
              'Cholesterol', 'Fatty acids, total saturated']

for c in nut_cols:
    if c not in snack.columns:
        raise ValueError(f"Snack nutrient column missing: {c}")
for c in ingnut_cols:
    if c not in ingnut_df.columns:
        raise ValueError(f"Ingnut nutrient column missing: {c}")

# ─── Resolve variants with separate base and case columns, then build indices ───
base_col_dyn = f"Snack_Top{args.top_n}_Ing"
base_col = base_col_dyn if base_col_dyn in ingnut_df.columns else (
    "Snack_Top200_Ing" if "Snack_Top200_Ing" in ingnut_df.columns else None
)
case_col = "case"
if base_col is None or case_col not in ingnut_df.columns:
    raise ValueError(f"Expected '{base_col_dyn}' or 'Snack_Top200_Ing' AND 'case' in ingnut_df.")

# unique key per variant row (avoid " | nan")
base_clean = ingnut_df[base_col].astype(str).str.strip()

case_raw = ingnut_df[case_col]
case_clean = case_raw.astype(object).where(pd.notna(case_raw), "")  # NaN -> ""
case_clean = case_clean.astype(str).str.strip()
# sometimes "nan"/"None" arrive as strings; normalize to empty
case_clean = case_clean.replace({"nan": "", "NaN": "", "None": "", "NONE": ""})

ingnut_df["variant_key"] = np.where(
    case_clean.eq(""),
    base_clean,
    base_clean + " | " + case_clean
)

# group variants by base name
ingnut_df["_base_norm"] = ingnut_df[base_col].str.upper().str.strip()
base_groups = ingnut_df.groupby("_base_norm").indices  # base -> row idx (variants)

# matrices for similarity / fit
X_all = ingnut_df[ingnut_cols].to_numpy(dtype=float)
X_norms = np.maximum(norm(X_all, axis=1), 1e-12)
Y_obs = snack[nut_cols].to_numpy(dtype=float)

def choose_cosine(base_token: str, y_vec: np.ndarray) -> str:
    base = str(base_token).upper().strip()
    if base not in base_groups: return base_token
    idxs = base_groups[base]
    y_n = max(norm(y_vec), 1e-12)
    sims = (X_all[idxs] @ y_vec) / (X_norms[idxs] * y_n)
    best_idx = idxs[int(np.argmax(sims))]
    return ingnut_df.loc[best_idx, "variant_key"]

def fit_residual_onecol(x_vec: np.ndarray, y_vec: np.ndarray) -> float:
    b = cp.Variable(1, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x_vec.reshape(-1,1) @ b - y_vec)),
                      [b <= 1])
    try:
        prob.solve(solver=solver_enum, **solver_kwargs)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return np.inf
        return float(np.sqrt(max(prob.value, 0.0)))
    except Exception:
        return np.inf

def choose_fit(base_token: str, y_vec: np.ndarray) -> str:
    base = str(base_token).upper().strip()
    if base not in base_groups: return base_token
    idxs = base_groups[base]
    best_idx, best_res = None, np.inf
    for idx in idxs:
        res = fit_residual_onecol(X_all[idx], y_vec)
        if res < best_res:
            best_idx, best_res = idx, res
    return ingnut_df.loc[best_idx, "variant_key"]

def choose_rule(base_token: str, snack_row: pd.Series, y_vec: np.ndarray) -> str:
    base = str(base_token).upper().strip()
    if base not in base_groups: return base_token
    idxs = base_groups[base]

    # MILK, NONFAT, DRIED -> prefer "without" (no vitamins)
    if base == "MILK, NONFAT, DRIED":
        cases = ingnut_df.loc[idxs, case_col].astype(str).str.lower()
        mask_without = cases.str.contains("without")
        if mask_without.any():
            chosen_idx = idxs[np.where(mask_without.to_numpy())[0][0]]
            return ingnut_df.loc[chosen_idx, "variant_key"]
        return choose_cosine(base_token, y_vec)

    # BUTTER -> salted vs unsalted via sodium proximity
    if base == "BUTTER":
        snack_na = float(snack_row.get('Sodium(mg)', np.nan))
        cand_na = ingnut_df.loc[idxs, 'Sodium, Na'].astype(float).to_numpy()
        if np.isnan(snack_na) or np.isnan(cand_na).any():
            return choose_cosine(base_token, y_vec)
        chosen_local = int(np.argmin(np.abs(cand_na - snack_na)))
        return ingnut_df.loc[idxs[chosen_local], "variant_key"]

    # default: cosine
    return choose_cosine(base_token, y_vec)

# resolve base tokens -> variant_key
resolved_lists = []
for j, ing_list in enumerate(snack['mapped_list_top20']):
    yj = Y_obs[j]
    row = snack.loc[j]
    out = []
    for base_token in ing_list:
        if (ingnut_df["variant_key"] == base_token).any():
            out.append(base_token); continue
        if args.resolve_mode == "rule":
            out.append(choose_rule(base_token, row, yj))
        elif args.resolve_mode == "fit":
            out.append(choose_fit(base_token, yj))
        else:
            out.append(choose_cosine(base_token, yj))
    resolved_lists.append(out)

snack['mapped_list_top20_resolved'] = resolved_lists

# ---- log BUTTER mapping counts (case 1 vs 2) ----------------------------------
# maps variant_key -> base/case for robust parsing
_key_to_base = ingnut_df.set_index("variant_key")[base_col].astype(str).str.upper().str.strip().to_dict()
_key_to_case = ingnut_df.set_index("variant_key")[case_col].astype(str).str.strip().to_dict()

butter_rows = []
for i, lst in enumerate(snack['mapped_list_top20_resolved']):
    for k in lst:
        base = _key_to_base.get(k, "")
        if base == "BUTTER":
            case = _key_to_case.get(k, "")
            butter_rows.append({"snack_index": i, "variant_key": k, "case": case})

butter_df = pd.DataFrame(butter_rows)
if not butter_df.empty:
    # case_id is "1" for "1(with salt)", "2" for "2(without salt)", else "other"
    butter_df["case_id"] = butter_df["case"].str.extract(r"^(\d+)").fillna("other")
    case_counts = butter_df["case_id"].value_counts().to_dict()
    n1 = int(case_counts.get("1", 0))
    n2 = int(case_counts.get("2", 0))
    nother = int(case_counts.get("other", 0))
    logger.info("BUTTER mapping counts → case 1(with salt)=%d, case 2(without salt)=%d, other=%d", n1, n2, nother)

# Build index universe over variant_key
variant_universe = ingnut_df["variant_key"].tolist()
ing_to_idx = {k:i for i, k in enumerate(variant_universe)}
snack[f'mapped_list_top{args.top_n}_only'] = snack['mapped_list_top20_resolved'].apply(
    lambda lst: [k for k in lst if k in ing_to_idx]
)
snack['Sj_indices'] = snack[f'mapped_list_top{args.top_n}_only'].apply(
    lambda lst: [ing_to_idx[k] for k in lst]
)

# X_full consistent with variant_universe order
X_full = ingnut_df.set_index("variant_key").loc[variant_universe, ingnut_cols].to_numpy().T
logger.info("X_full shape: %s", X_full.shape)  # (8, n_variants)

# Extra nutrients to predict via inferred weights
extra_ingnut_cols_all = ['Calcium, Ca', 'Fiber, total dietary', 'Iron, Fe']
extra_ingnut_cols = [c for c in extra_ingnut_cols_all if c in ingnut_df.columns]

X_extra = None
if extra_ingnut_cols:
    X_extra = ingnut_df.set_index("variant_key") \
        .loc[variant_universe, extra_ingnut_cols] \
        .to_numpy(dtype=float).T  # shape: (q_extra, K)
    logger.info("X_extra shape: %s for extra nutrients %s", X_extra.shape, extra_ingnut_cols)
else:
    logger.info("No extra nutrients found in ingnut_df among %s", extra_ingnut_cols_all)


# ─── 2) build Y_mat ────────────────────────────────────────────────────────────
Y_mat = snack[nut_cols].to_numpy()
n_snack, p = Y_mat.shape
logger.info("Y_mat shape: %s", Y_mat.shape)


# ─── 3) scale with residual weight function ─────────────────────────────────
def make_residual_weights(X_full, Y_mat, mode="logiqr"):
    """
    Returns weights w (length p) so the loss is || diag(w) (X_j b - y_j) ||^2.

    Modes:
      - none:    no weighting
      - std:     1 / std
      - mad:     1 / (1.4826 * MAD)       [with safety floor]
      - range:   1 / (max - min)
      - minmax:  alias of range
      - pminmax: 1 / (p99 - p1)           [robust to outliers]
      - iqr:     1 / (Q3 - Q1)
      - logstd:  1 / std(log1p(values))
      - logiqr:  1 / (Q3 - Q1) on log1p(values)  ← recommended default

    All modes are normalized so mean(w) = 1, with safe clipping.
    """
    EPS = 1e-9
    # stack to capture both ingredient & snack scales
    A = np.vstack([X_full.T, Y_mat])  # shape (K+n, p)

    if mode == "none":
        w = np.ones(A.shape[1])

    elif mode == "std":
        s = np.nanstd(A, axis=0, ddof=1)
        w = 1.0 / np.clip(s, EPS, None)

    elif mode == "mad":
        med = np.nanmedian(A, axis=0)
        mad = np.nanmedian(np.abs(A - med), axis=0)
        s = 1.4826 * mad
        # floor MAD using 5% of robust p1–p99 range to avoid huge weights
        p1, p99 = np.nanpercentile(A, [1, 99], axis=0)
        robust_rng = np.clip(p99 - p1, EPS, None)
        s = np.maximum(s, 0.05 * robust_rng)
        w = 1.0 / np.clip(s, EPS, None)

    elif mode in ("range", "minmax"):
        s = np.nanmax(A, axis=0) - np.nanmin(A, axis=0)
        w = 1.0 / np.clip(s, EPS, None)

    elif mode == "pminmax":
        low = np.nanpercentile(A, 1.0, axis=0)
        high = np.nanpercentile(A, 99.0, axis=0)
        s = high - low
        w = 1.0 / np.clip(s, EPS, None)

    elif mode == "iqr":
        q1 = np.nanpercentile(A, 25.0, axis=0)
        q3 = np.nanpercentile(A, 75.0, axis=0)
        s = q3 - q1
        w = 1.0 / np.clip(s, EPS, None)

    elif mode == "logstd":
        A_pos = np.where(A > 0, A, 0.0)
        s = np.nanstd(np.log1p(A_pos), axis=0, ddof=1)
        w = 1.0 / np.clip(s, EPS, None)

    elif mode == "logiqr":
        # robust spread on log scale (best for heavy-tailed, nonnegative nutrients)
        A_pos = np.where(A > 0, A, 0.0)  # guard tiny negatives if any
        A_log = np.log1p(A_pos)
        q1, q3 = np.nanpercentile(A_log, [25.0, 75.0], axis=0)
        s = q3 - q1
        # avoid degenerate columns
        w = 1.0 / np.clip(s, EPS, None)
    else:
        w = np.ones(A.shape[1])
    # stabilize & normalize weights
    w = np.nan_to_num(w, nan=1.0, posinf=1e6, neginf=1e-6)
    # clip to avoid pathological extremes but keep dynamic range
    w = np.clip(w, 1e-6, 1e6)
    # normalize so mean weight = 1
    mu = np.mean(w) if np.isfinite(np.mean(w)) and np.mean(w) > 0 else 1.0
    w = w / mu
    return w.astype(float)

W_vec = make_residual_weights(X_full, Y_mat, mode=args.scale)  # length p
W = np.diag(W_vec)  # (p x p)

# ─── 4) deconvolution function ─────────────────────────────────────────────────
def solve_osqp(X_full, Y_mat, S_indices, solver_enum, solver_kwargs,
               constraint_mode="nnls_only", W_mat=None, robust=False):
    """
    constraint_mode:
      - "nnls_only": b >= 0,
      - "nnls_mono": b >= 0, b[k] >= b[k+1]
      - "eq1":       b >= 0, sum(b) == 1
      - "eq1_mono":  b >= 0, sum(b) == 1,    b[k] >= b[k+1]
      - "le1":       b >= 0, sum(b) <= 1
      - "le1_mono":  b >= 0, sum(b) <= 1,    b[k] >= b[k+1]
    The objective is || W (X_j b - y_j) ||^2 (Huber if robust=True).
    """
    n_snack, p = Y_mat.shape
    K = X_full.shape[1]
    preds = np.zeros((n_snack, K))
    failed = []
    W = W_mat if W_mat is not None else np.eye(p)

    def loss(expr):
        return cp.sum_squares(expr) if not robust else cp.sum(cp.huber(expr, 1.0))

    for j, Sj in enumerate(S_indices):
        if not Sj:
            continue

        Kj = len(Sj)
        Xj = X_full[:, Sj]
        var = cp.Variable(Kj, nonneg=True)

        cons = []
        if constraint_mode in ("eq1", "eq1_mono"):
            cons.append(cp.sum(var) == 1)
        elif constraint_mode in ("le1", "le1_mono"):
            cons.append(cp.sum(var) <= 1)
        elif constraint_mode in ("nnls_only", "nnls_mono"):
            pass
        else:
            raise ValueError(f"Unknown constraint mode: {constraint_mode}")

        if constraint_mode.endswith("_mono") and Kj > 1:
            cons += [var[k] >= var[k+1] for k in range(Kj - 1)]

        resid = (W @ Xj) @ var - (W @ Y_mat[j])
        prob = cp.Problem(cp.Minimize(loss(resid)), cons)

        try:
            prob.solve(solver=solver_enum, **solver_kwargs)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                raise SolverError(f"status={prob.status}")
            wj = var.value
        except Exception:
            failed.append(j)
            wj = np.ones(Kj) / Kj  # fallback

        full = np.zeros(K); full[Sj] = wj
        preds[j] = full

    return preds, failed


# ─── 5) evaluation metric ─────────────────────────────────────────────────────
def evaluate(Y_true, Y_pred):
    mse = mean_squared_error(Y_true.flatten(), Y_pred.flatten())
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_true.flatten(), Y_pred.flatten())
    return rmse, r2

# ─── 6) hyperparameter tuning ─────────────────────────────────────────────────

if args.tune and solver_key != "osqp":
    logger.warning("Tuning grid is implemented for OSQP only. Running a single solve with %s.", solver_key)
    args.tune = False

if args.tune:
    eps_abs_list = [1e-3, 1e-4, 1e-5]
    eps_rel_list = [1e-3, 1e-4, 1e-5]
    max_iter_list = [100_000, 200_000, 300_000]
    rho_list      = [1.0, 0.1, 0.01]

    grid = [
        {"eps_abs": ea, "eps_rel": er, "max_iter": mi, "rho": r}
        for ea, er, mi, r in itertools.product(
            eps_abs_list, eps_rel_list, max_iter_list, rho_list
        )
    ]
    results = []
    for opts in grid:
        logger.info("Tuning with OSQP opts: %s", opts)
        preds, failed = solve_osqp(X_full, Y_mat, snack['Sj_indices'].tolist(), solver_enum, solver_kwargs,
                                    constraint_mode=args.constraint, W_mat=W, robust=args.robust)
        Y_pred = preds @ X_full.T
        rmse, r2 = evaluate(Y_mat, Y_pred)
        results.append({**opts, 'rmse':rmse, 'r2':r2, 'fail_rate':len(failed)/n_snack})
    df_tune = pd.DataFrame(results)
    df_tune.to_csv(f"../outputs/{args.output_prefix}_osqp_tuning_results.csv", index=False)
    print(df_tune)
    best = df_tune.loc[df_tune['rmse'].idxmin()]
    logger.info("Best params: %s", best.to_dict())
    best_opts = {k:best[k] for k in grid[0].keys()}
    preds, failed = solve_osqp(
        X_full, Y_mat, snack['Sj_indices'].tolist(), best_opts, constraint_mode=args.constraint
    )
else:
    default_opts = {'eps_abs':1e-5,'eps_rel':1e-5,'max_iter':200_000}
    preds, failed = solve_osqp(
        X_full, Y_mat, snack['Sj_indices'].tolist(), solver_enum, default_opts, constraint_mode=args.constraint, W_mat=W, robust=args.robust
    )

failed_set = set(failed)
fail_count = len(failed_set)
n_total = len(snack)
n_with_S = snack['Sj_indices'].apply(lambda x: len(x) > 0).sum()
success_idxs = [j for j, Sj in enumerate(snack['Sj_indices']) if len(Sj) > 0 and j not in failed_set]
success_count = len(success_idxs)
fail_count = len(failed)
skipped_count = n_total - n_with_S  # snacks with empty S_j

logger.info(
    f"{args.solver} results: success=%d, failed=%d (%.2f%%), skipped(no S_j)=%d, total=%d",
    success_count, fail_count, (fail_count / max(n_with_S, 1)) * 100.0, skipped_count, n_total
)

# ─── 7) save predictions ────────────────────────────────────────────────────────
np.save(f"../outputs/{args.output_prefix}_preds.npy", preds)
pred_df = pd.DataFrame(preds, columns=variant_universe)
pred_df.to_csv(f"../outputs/{args.output_prefix}_preds.csv", index=False)

# ─── 8) compute & save reconstruction metrics ──────────────────────────────────
Y_pred = preds @ X_full.T
rmse, r2 = evaluate(Y_mat, Y_pred)
metrics = pd.DataFrame([{'rmse':rmse,'r2':r2,'fail_rate':len(failed)/n_snack}])
metrics.to_csv(f"../outputs/{args.output_prefix}_metrics.csv", index=False)

# Predict extra nutrients via the same weights
extra_pred_cols = []
pred_name_map = {  # output column names
    'Calcium, Ca': 'pred_Calcium(mg)',
    'Fiber, total dietary': 'pred_Fiber(g)',
    'Iron, Fe': 'pred_Iron(mg)'
}

if X_extra is not None:
    Y_extra_pred = preds @ X_extra.T  # (n_snack × q_extra)
    for i, col in enumerate(extra_ingnut_cols):
        snack[pred_name_map[col]] = Y_extra_pred[:, i]
        extra_pred_cols.append(pred_name_map[col])


# ─── Build plotting/output df with side-by-side nutrient columns ───────────────
failed_set = set(failed)

# attach predictions by column
pred_nutrient_cols = [f"pred_{c}" for c in nut_cols]
snack[pred_nutrient_cols] = Y_pred

# status + weights
snack["failed"] = snack.index.isin(failed_set).astype(int)
snack["ing_weights"] = pred_df.apply(
    lambda row: [row.iloc[i] for i in snack.at[row.name, "Sj_indices"]],
    axis=1
)
snack["ing_weights_clipped"] = pred_df.apply(
    lambda row: np.clip([row.iloc[i] for i in snack.at[row.name, "Sj_indices"]], 0, 1).tolist(),
    axis=1
)

# Mapping stats & type 
snack['n_mapped_ing'] = snack[f'mapped_list_top{args.top_n}_only'].apply(len)
snack['n_total_ing'] = snack['mapped_list_top20'].apply(len)
snack['mapped_ratio'] = snack['n_mapped_ing'] / snack['n_total_ing']

# compute `first_mapped`: whether the 1st original ingredient got mapped
def _first_mapped(row):
    lst = row['mapped_list_top20']
    if not isinstance(lst, list) or len(lst) == 0:
        return False
    first_base = str(lst[0]).upper().strip()
    mapped_vars = row[f"mapped_list_top{args.top_n}_only"]  # variant_keys kept
    # _key_to_base was created earlier (variant_key -> BASE normalized)
    mapped_bases = {_key_to_base.get(k, "") for k in mapped_vars}
    return first_base in mapped_bases

snack["first_mapped"] = snack.apply(_first_mapped, axis=1)

def _mapping_type(row):
    if row["n_mapped_ing"] == 0:
        return "none"
    if pd.isna(row["mapped_ratio"]) or row["mapped_ratio"] < 0.7:
        return "low mapped ratio (<70%)"
    return "good (≥70%)"

snack["mapping_type"] = snack.apply(_mapping_type, axis=1)

# reorder columns so each nutrient sits next to its prediction
pair_cols = [c for c in nut_cols for c in (c, f"pred_{c}")]
pair_set = set(pair_cols)

# add extra nutrient pairs (only add the true column if it exists in snack)
extra_pairs = []
if X_extra is not None:
    for col in extra_ingnut_cols:
        true_col = col
        pred_col = pred_name_map[col]
        if true_col in snack.columns:
            extra_pairs.extend([true_col, pred_col])
        else:
            extra_pairs.extend([pred_col])  # only prediction available

pair_cols = pair_cols + extra_pairs
pair_set = set(pair_cols)

other_cols = [c for c in snack.columns if c not in pair_set]  # keep original order for the rest
snack = snack[other_cols + pair_cols]

# save
snack.to_csv(f"../outputs/{args.output_prefix}_snack_output.csv", index=False)

# ─── 9) plotting (color: mapping_type, marker: first_mapped) ───────────────────
from matplotlib.lines import Line2D


base_x = [
    "Energy(kcal)", "Carbohydrate(g)", "Total fat(g)",
    "Protein(g)", "Total sugar(g)", "Sodium(mg)",
    "Cholesterol(mg)", "Saturated fatty acids(g)"
]
extra_x = []
if "Calcium(mg)" in snack.columns and "pred_Calcium(mg)" in snack.columns:
    extra_x.append("Calcium(mg)")
if "Fiber(g)" in snack.columns and "pred_Fiber(g)" in snack.columns:
    extra_x.append("Fiber(g)")
if "Iron(mg)" in snack.columns and "pred_Iron(mg)" in snack.columns:
    extra_x.append("Iron(mg)")

x_cols = [c for c in base_x + extra_x if f"pred_{c}" in snack.columns]
y_cols = [f"pred_{c}" for c in x_cols]

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
alpha = 0.1 if not (args.n_sample is not None and args.n_sample < 1000) else 1.0

# canonical plotting categories (match _map_type_key)
key_order = ["good", "low_ratio", "none"]
key_label = {
    "good":      "good (≥70%)",
    "low_ratio": "low mapped ratio (<70%)",
    "none":      "none",
}
key_color = {"good": "C2", "low_ratio": "C1", "none": "C0"}
marker_map = {True: "o", False: "x"}

def _norm_map_type(s: str) -> str:
    if not isinstance(s, str):
        return "none"
    t = s.strip().lower().replace("≥", ">=").replace("  ", " ")
    if "good" in t:                         # e.g. "good (>=70%, first mapped)"
        return "good"
    if "first_unmapped" in t:
        return "first_unmapped"
    if "low mapped ratio" in t or "<70" in t:
        return "low_ratio"
    if "none" in t:
        return "none"
    # fallback: try to infer
    if "first" in t and "map" in t and "un" in t:
        return "first_unmapped"
    return "none"

snack["_map_type_key"] = snack["mapping_type"].apply(_norm_map_type)

for ax, x, y in zip(axes, x_cols, y_cols):
    if x not in snack.columns or y not in snack.columns:
        ax.axis("off"); continue

    xs = snack[x].to_numpy(dtype=float)
    ys = snack[y].to_numpy(dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)

    for key in key_order:
        for fm in (True, False):
            m = (snack["_map_type_key"] == key) & (snack["first_mapped"] == fm) & valid
            if m.any():
                ax.scatter(xs[m], ys[m], s=12, alpha=alpha,
                           color=key_color[key], marker=marker_map[fm])

    if valid.any():
        mn = float(np.nanmin([xs[valid].min(), ys[valid].min()]))
        mx = float(np.nanmax([xs[valid].max(), ys[valid].max()]))
        if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
            mn, mx = 0.0, 1.0
    else:
        mn, mx = 0.0, 1.0

    ax.plot([mn, mx], [mn, mx], ls="--", c="gray", lw=1)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)

    try:
        r2 = r2_score(xs[valid], ys[valid]) if valid.any() else np.nan
        rmse = np.sqrt(mean_squared_error(xs[valid], ys[valid])) if valid.any() else np.nan
        ax.text(0.03, 0.97, f"R²={r2:.3f}\nRMSE={rmse:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9)
    except Exception:
        pass

    ax.set_title(f"{x} vs {y}")
    ax.set_xlabel(x); ax.set_ylabel(y)

# turn off any unused subplots
for k in range(len(x_cols), len(axes)):
    axes[k].axis("off")

# legends: only show categories actually present
present_keys = [k for k in key_order if (snack["_map_type_key"] == k).any()]
type_handles = [Patch(facecolor=key_color[k], label=key_label[k], alpha=0.9) for k in present_keys]
first_handles = [
    Line2D([0], [0], marker=marker_map[True], color="none", markerfacecolor="black",
           markersize=7, label="first mapped"),
    Line2D([0], [0], marker=marker_map[False], color="black", linestyle="",
           markersize=7, label="first NOT mapped"),
]
fig.legend(handles=type_handles, title="Mapping type", loc="upper right")
fig.legend(handles=first_handles, title="First ingredient", loc="lower right")

plt.suptitle(
    f"True vs Predicted Nutrients ({args.solver.upper()})\n"
    f"scale={args.scale}, resolve={args.resolve_mode}, constraint={args.constraint}, N={len(snack)}"
)
plt.tight_layout()
plt.savefig("../outputs/" + args.output_prefix + "_true_vs_predicted_plot.png",
            dpi=300, bbox_inches="tight")

# end
print("Done. Outputs under:", args.output_prefix)