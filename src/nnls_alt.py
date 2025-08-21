# src/nnls_alt.py

import os
import argparse
import sys
import ast
import numpy as np
import pandas as pd
import logging
from time import time
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error, r2_score

# ─── Argument Parsing ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NNLS-based deconvolution baseline")
parser.add_argument("--workdir", "-C", type=str, default=None,
                    help="Working directory override")
parser.add_argument("--snack-data", "-S", type=str, required=True,
                    help="Path to snack data CSV")
parser.add_argument("--ing-nut-data", "-I", type=str, required=True,
                    help="Path to ingredient–nutrient data CSV")
parser.add_argument("--output-prefix", "-O", type=str, required=True,
                    help="Prefix for output files")
parser.add_argument("--top-n", type=int, required=True,
                    help="Number of top ingredients to use")
args = parser.parse_args()

if args.workdir:
    os.chdir(os.path.expanduser(args.workdir))
else:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─── Load data ─────────────────────────────────────────────────────────────────
snack = pd.read_csv(args.snack_data)
ingnut_df = pd.read_csv(args.ing_nut_data)
snack['mapped_list_top20'] = snack['mapped_list_top20'].apply(ast.literal_eval)

top_n = ingnut_df[f"Snack_Top200_Ing"].tolist()
ing_to_idx = {ing:i for i,ing in enumerate(top_n)}
snack['Sj_indices'] = snack['mapped_list_top20'].apply(
    lambda lst: [ing_to_idx[ing] for ing in lst if ing in ing_to_idx]
)

nut_cols = ['Energy(kcal)', 'Total fat(g)', 'Protein(g)',
            'Carbohydrate(g)', 'Total sugar(g)', 'Sodium(mg)',
            'Cholesterol(mg)', 'Saturated fatty acids(g)']
Y_mat = snack[nut_cols].to_numpy()

ingnut_cols = [
    'Energy', 'Total lipid (fat)', 'Protein',
    'Carbohydrate, by difference', 'Sugars, total', 'Sodium, Na',
    'Cholesterol', 'Fatty acids, total saturated'
]

X_full = ingnut_df[ingnut_cols].to_numpy().T

# ─── Deconvolution via NNLS ────────────────────────────────────────────────────
n_snack, _ = Y_mat.shape
K = len(top_n)
preds = np.zeros((n_snack, K))
failed = []

for j in range(n_snack):
    Sj = snack.at[j, 'Sj_indices']
    if not Sj:
        continue
    Xj = X_full[:, Sj]
    yj = Y_mat[j]
    bj, _ = nnls(Xj, yj)
    if bj.sum() > 0:
        bj = bj / bj.sum()
    else:
        bj = np.ones(len(Sj)) / len(Sj)
        failed.append(j)
    full_b = np.zeros(K)
    full_b[Sj] = bj
    preds[j] = full_b

# ─── Evaluate & save ───────────────────────────────────────────────────────────
Y_pred = preds @ X_full.T
rmse = np.sqrt(mean_squared_error(Y_mat.flatten(), Y_pred.flatten()))
r2 = r2_score(Y_mat.flatten(), Y_pred.flatten())

pd.DataFrame(preds, columns=top_n).to_csv(f"{args.output_prefix}_nnls_preds.csv", index=False)
pd.DataFrame([{'rmse':rmse,'r2':r2,'fail_rate':len(failed)/n_snack}]) \
    .to_csv(f"{args.output_prefix}_nnls_metrics.csv", index=False)

logger.info("NNLS RMSE=%.4f, R2=%.4f, fail_rate=%.2f%%",
            rmse, r2, len(failed)/n_snack*100)
