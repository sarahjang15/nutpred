import pandas as pd
import numpy as np
import ast
import os
from typing import Tuple, Iterable, Optional

import nutpred.constants as C


def compute_zero_thresholds(
    input_csv: str,
    zero_threshold_csv: str,
    output_csv: Optional[str] = None,
    nutrients: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Args:
        snack_change_to_zero_csv: Path to CSV like 'data/snack_change_to_zero.csv'.
        zero_threshold_csv: Path to CSV like 'data/zero_nut_threshold.csv'.

    Behavior:
        - Computes threshold-per-100g values for specified nutrients (or all in C.NUT11 if not provided).
        - Returns and saves a DataFrame with the EXACT SAME columns as input_csv,
          where specified nutrient columns are updated ONLY where a threshold applies
          (i.e., serving_size present and original nutrient value is 0). Otherwise values are unchanged.
        - Saves a separate thresholds-only DataFrame to the data folder (same folder as input by default).

    Returns:
        DataFrame with identifier column(s) and per-nutrient threshold columns named as nutrients.
    """
    zero = pd.read_csv(input_csv)
    thr = pd.read_csv(zero_threshold_csv)

    def get_zero_thres(row: pd.Series, nut: str):
        if pd.notna(row.get("serving_size")):
            if row.get(nut) == 0:
                # threshold value is per serving; convert to per-100g using serving_size
                try:
                    base = thr[thr["nutrient"] == nut]["zero_threshold_value"].values[0]
                except Exception:
                    return np.nan
                return base / 2  * 100 / row["serving_size"]
        return np.nan

    # Create an output DataFrame with the exact same columns as input
    df_out = zero.copy()

    # Determine which nutrients to process
    target_nutrients = list(nutrients) if nutrients is not None else list(C.NUT11)
    for nut in target_nutrients:
        thres_series = zero.apply(get_zero_thres, axis=1, nut=nut)
        # Update only where threshold is available; keep original otherwise
        df_out[nut] = np.where(pd.notna(thres_series), thres_series, df_out.get(nut))

    # Determine default output path if not provided (same folder as input CSV)
    if output_csv is None:
        out_dir = os.path.dirname(os.path.abspath(input_csv))
        base = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = os.path.join(out_dir, f"{base}_thresholded.csv")

    df_out.to_csv(output_csv, index=False)
    return df_out


def build_snack_input_df_new(
    snack_input_df_csv: str,
    snack_umap_added_csv: str,
    snack_change_to_zero_csv: str,
) -> pd.DataFrame:
    """
    Build snack_input_df_new by concatenating selected columns from zero_out with UMAP-added features.

    Mirrors notebook cells that formed zero_final and concatenated with 'snack_umap_added.csv'.
    """
    snack_input_df = pd.read_csv(snack_input_df_csv)
    zero_df = pd.read_csv(snack_change_to_zero_csv)

    # zero_final takes fdc_id + all columns of snack_input_df except the last 4 (as in notebook)
    keep_cols = ["fdc_id"] + snack_input_df.columns.tolist()[:-4]
    keep_cols = [c for c in keep_cols if c in zero_df.columns]
    zero_final = zero_df[keep_cols].copy()

    snack_umap_added = pd.read_csv(snack_umap_added_csv)
    snack_input_new = pd.concat([zero_final, snack_umap_added], axis=1)
    return snack_input_new


def make_full_df_nona(snack_input_new: pd.DataFrame) -> pd.DataFrame:
    """
    Create a non-null dataframe for modeling by dropping rows with NaN on required nutrients.

    This follows the intention of "full_df_nona" in the notebook context, ensuring that
    all NUT11 nutrient columns exist and are non-null.
    """
    required_cols = [c for c in C.NUT11 if c in snack_input_new.columns]
    if not required_cols:
        # If none of the required columns exist yet, return as-is
        return snack_input_new.copy()
    full_df_nona = snack_input_new.dropna(subset=required_cols).reset_index(drop=True)
    return full_df_nona


def prepare_until_full_df_nona(
    snack_change_to_zero_csv: str,
    zero_threshold_csv: str,
    snack_input_df_csv: str,
    snack_umap_added_csv: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience orchestrator that runs the steps up to creating full_df_nona.

    Returns:
        zero_out_df, snack_input_new_df, full_df_nona_df
    """
    zero_out_df = compute_zero_thresholds(snack_change_to_zero_csv, zero_threshold_csv)
    snack_input_new_df = build_snack_input_df_new(
        snack_input_df_csv, snack_umap_added_csv, snack_change_to_zero_csv
    )
    full_df_nona_df = make_full_df_nona(snack_input_new_df)
    return zero_out_df, snack_input_new_df, full_df_nona_df


def embed_ingredients_and_save(
    input_csv: str,
    output_csv: str,
    api_key: Optional[str] = None,
    model: str = "text-embedding-3-small",
    prompt_prefix: str = (
        "Ingredients listed in descending order by quantity (from most to least); "
        "parentheses indicate sub-ingredients: "
    ),
) -> pd.DataFrame:
    """
    Generate OpenAI embeddings for the ingredients using a prompted string and save to CSV.

    Reads input CSV with an 'ingredients' column and writes a CSV that adds:
      - ingredients_prompt
      - ingredients_embedding_added (list[float])

    Args:
        input_csv: Source CSV path
        output_csv: Output CSV path to write with added columns
        api_key: OpenAI API key; if None, will use environment var OPENAI_API_KEY
        model: OpenAI embedding model name
        prompt_prefix: Text prepended before the ingredients to form the prompt
    """
    # Lazy import to avoid import-time dependency errors
    try:
        from openai import OpenAI  # type: ignore
    except Exception as _e:
        raise ImportError("openai package is required for embedding. Install and retry.")

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
    if not client.api_key:
        raise ValueError("OpenAI API key not provided. Pass api_key or set OPENAI_API_KEY env var.")

    df = pd.read_csv(input_csv)
    if "ingredients" not in df.columns:
        raise ValueError("Input CSV must contain an 'ingredients' column")

    def make_prompt(text: str) -> str:
        text = text if isinstance(text, str) else ""
        return f"{prompt_prefix}{text}"

    df = df.copy()
    df["ingredients_prompt"] = df["ingredients"].fillna("").apply(make_prompt)

    def get_embedding_safe(text: str):
        if not text:
            return None
        try:
            response = client.embeddings.create(model=model, input=text)
            return response.data[0].embedding
        except Exception:
            return None

    df["ingredients_embedding_added"] = df["ingredients_prompt"].apply(get_embedding_safe)
    df.to_csv(output_csv, index=False)
    return df


def compute_umap_from_embeddings(
    input_csv: str,
    output_csv: str,
    n_components_list: Iterable[int] = (2, 5, 10, 20),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute UMAP coordinates from the 'ingredients_embedding_added' column and save to CSV.

    The output CSV contains one list column per n_components value, named 'umap_{n}',
    and an 'fdc_id' column copied from the input (if present).
    """
    # Lazy import to avoid import-time dependency errors
    try:
        from umap import UMAP  # type: ignore
    except Exception as _e:
        raise ImportError("umap-learn is required for UMAP. Install with `pip install umap-learn`. ")

    df = pd.read_csv(input_csv)
    if "ingredients_embedding_added" not in df.columns:
        raise ValueError("Input CSV must contain 'ingredients_embedding_added' column")

    # Parse stringified embeddings
    embeddings = df["ingredients_embedding_added"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).tolist()

    result = pd.DataFrame(index=df.index)
    for n in n_components_list:
        reducer = UMAP(n_components=int(n), random_state=random_state)
        coords = reducer.fit_transform(embeddings)
        result[f"umap_{int(n)}"] = coords.tolist()

    if "fdc_id" in df.columns:
        result["fdc_id"] = df["fdc_id"]

    result.to_csv(output_csv, index=False)
    return result


def filter_valid_nutrition_data(df):
    """
    Filter out rows with invalid nutrition data based on validation rules.
    
    Returns:
        DataFrame: Filtered DataFrame with only valid rows
    """
    valid_indices = []
    
    for idx, row in df.iterrows():
        is_valid = True
        
        # 1) 에너지가 > 1,080 kcal 초과인 경우
        if pd.notna(row["Energy(kcal)"]) and row["Energy(kcal)"] >= 0:
            if row["Energy(kcal)"] > 1080:
                is_valid = False
                continue
        
        # 2) 에너지가 (탄수화물, 단백질, 지방으로 산출한 에너지) 20% 초과인 경우
        if ((pd.notna(row["Energy(kcal)"]) and row["Energy(kcal)"] >= 0) and 
            (pd.notna(row["Carbohydrate(g)"]) and row["Carbohydrate(g)"] >= 0) and
            (pd.notna(row["Protein(g)"]) and row["Protein(g)"] >= 0) and
            (pd.notna(row["Total fat(g)"]) and row["Total fat(g)"] >= 0)):
            calculated_energy = (
                (row["Carbohydrate(g)"] if pd.notna(row["Carbohydrate(g)"]) else 0) * 4 +
                (row["Total fat(g)"] if pd.notna(row["Total fat(g)"]) else 0) * 9 +
                (row["Protein(g)"] if pd.notna(row["Protein(g)"]) else 0) * 4
            )
            lower_bound = calculated_energy * 0.8
            upper_bound = calculated_energy * 1.2
            if not (lower_bound <= row["Energy(kcal)"] <= upper_bound):
                is_valid = False
                continue
        
        # 3) 지방, 포화지방, 트랜스지방 이상치 검토
        # 3-1) 포화지방산 > 지방 인 경우
        if (pd.notna(row["Total fat(g)"]) and row["Total fat(g)"] >= 0) and (pd.notna(row["Saturated fatty acids(g)"]) and row["Saturated fatty acids(g)"] >= 0):
            if row["Total fat(g)"] < row["Saturated fatty acids(g)"]:
                is_valid = False
                continue
        
        # 3-2) 트랜스지방산 > 지방 인 경우
        if (pd.notna(row["Total fat(g)"]) and row["Total fat(g)"] >= 0) and (pd.notna(row["Trans fatty acids(g)"]) and row["Trans fatty acids(g)"] >= 0):
            if row["Total fat(g)"] < row["Trans fatty acids(g)"]:
                is_valid = False
                continue
        
        # 3-3) '지방' < ('포화지방산'+'트랜스지방산'+'콜레스테롤/1000') 인 경우
        if pd.notna(row["Total fat(g)"]) and row["Total fat(g)"] >= 0:
            trans_fat = row["Trans fatty acids(g)"] if pd.notna(row["Trans fatty acids(g)"]) else 0
            sat_fat = row["Saturated fatty acids(g)"] if pd.notna(row["Saturated fatty acids(g)"]) else 0
            cholesterol = row["Cholesterol(mg)"] / 1000 if pd.notna(row["Cholesterol(mg)"]) else 0
            
            total_fat = trans_fat + sat_fat + cholesterol
            
            if total_fat > row["Total fat(g)"]:
                is_valid = False
                continue
        
        # 4) 탄수화물, 지방, 단백질 이상치 검토: ('100g 당 탄수화물'+ '100g당 단백질'+ '100g당 지방') ≤ 100
        total_macros = sum([
            row["Carbohydrate(g)"] if pd.notna(row["Carbohydrate(g)"]) else 0,
            row["Total fat(g)"] if pd.notna(row["Total fat(g)"]) else 0,
            row["Protein(g)"] if pd.notna(row["Protein(g)"]) else 0
        ])
        
        if total_macros >= 100.0001:
            is_valid = False
            continue
        
        if is_valid:
            valid_indices.append(idx)
        
    df = df.loc[valid_indices]
    df.reset_index(drop=True, inplace=True)
    return df


def prepare_pipeline(
    raw_dir: str,
    prefer_sugars_including_nlea: bool = True,
) -> pd.DataFrame:
    """
    Build a standardized dataframe from FoodData Central raw CSVs.

    Required files in raw_dir:
      - food.csv
      - branded_food.csv
      - food_nutrient.csv
      - nutrient.csv

    Returns a dataframe with columns expected by this project, including:
      ['fdc_id','branded_food_category','brand_owner','market_country','description','ingredients']
      + C.NUT11 (Energy, macro/micro nutrients with units in names)
    """
    food_csv = os.path.join(raw_dir, "food.csv")
    branded_csv = os.path.join(raw_dir, "branded_food.csv")
    fn_csv = os.path.join(raw_dir, "food_nutrient.csv")
    nut_csv = os.path.join(raw_dir, "nutrient.csv")

    food = pd.read_csv(food_csv)
    branded = pd.read_csv(branded_csv)
    food_nutrient = pd.read_csv(fn_csv)
    nutrient = pd.read_csv(nut_csv)

    # Deduplicate branded by latest publication_date per gtin_upc, then merge by fdc_id
    branded_for_dedup = branded.copy()
    # Ensure publication_date is available (pull from food if needed)
    if "publication_date" not in branded_for_dedup.columns and "publication_date" in food.columns:
        branded_for_dedup = branded_for_dedup.merge(
            food[["fdc_id", "publication_date"]], on="fdc_id", how="left"
        )
    # Parse and sort by publication_date
    if "publication_date" in branded_for_dedup.columns:
        branded_for_dedup["_pub_dt"] = pd.to_datetime(branded_for_dedup["publication_date"], errors="coerce")
    else:
        # Fallback: use fdc_id ordering if no date
        branded_for_dedup["_pub_dt"] = pd.NaT
    # If gtin_upc exists, sort by publication_date then keep last per gtin_upc
    if "gtin_upc" in branded_for_dedup.columns:
        branded_for_dedup = branded_for_dedup.sort_values(["_pub_dt", "fdc_id"]).drop_duplicates(
            subset=["gtin_upc"], keep="last"
        )
    # Base metadata join (on fdc_id as required)
    base = food.merge(branded_for_dedup, on="fdc_id", how="inner")

    # Select metadata columns if present
    meta_cols = {}
    meta_cols["fdc_id"] = base["fdc_id"]
    meta_cols["description"] = base["description_x"] if "description_x" in base.columns else base.get("description", pd.Series(index=base.index))
    for col in ["branded_food_category", "brand_owner", "market_country", "ingredients"]:
        if col in base.columns:
            meta_cols[col] = base[col]
        else:
            meta_cols[col] = pd.Series([None] * len(base))
    meta_df = pd.DataFrame(meta_cols)

    # Build nutrient wide table
    # Map FDC nutrient names to our standardized column names (no sugar preference; use 'Total Sugars')
    fdc_to_target = {
        "Energy": "Energy(kcal)",
        "Protein": "Protein(g)",
        "Total lipid (fat)": "Total fat(g)",
        "Carbohydrate, by difference": "Carbohydrate(g)",
        "Fatty acids, total saturated": "Saturated fatty acids(g)",
        "Fatty acids, total trans": "Trans fatty acids(g)",
        "Cholesterol": "Cholesterol(mg)",
        "Sodium, Na": "Sodium(mg)",
        "Fiber, total dietary": "Fiber(g)",
        "Calcium, Ca": "Calcium(mg)",
        "Iron, Fe": "Iron(mg)",
        "Total Sugars": "Total sugar(g)",
        "Potassium, K": "Potassium(mg)",
        "Vitamin A, IU": "Vitamin A(IU)",
        "Vitamin A, RAE": "Vitamin A(RAE)",
        "Vitamin C, total ascorbic acid": "Vitamin C(mg)"
    }

    fn = food_nutrient.merge(nutrient[["id", "name"]], left_on="nutrient_id", right_on="id", how="left")
    wanted_fdc_names = set(fdc_to_target.keys())
    fn = fn[fn["name"].isin(wanted_fdc_names)].copy()
    fn["target_col"] = fn["name"].apply(lambda n: fdc_to_target.get(n, n))
    # Resolve duplicates (no priority needed when using a single sugar name)
    fn = fn.sort_values(["fdc_id", "target_col"]).drop_duplicates(
        subset=["fdc_id", "target_col"], keep="first"
    )

    wide = fn.pivot(index="fdc_id", columns="target_col", values="amount").reset_index()

    # Join metadata and nutrients
    full = meta_df.merge(wide, on="fdc_id", how="left")

    # Ensure all expected nutrient columns exist (even if missing)
    for col in C.NUT11 + ["Trans fatty acids(g)"]:
        if col not in full.columns:
            full[col] = np.nan

    # Final ordering
    ordered_cols = [
        "fdc_id", "branded_food_category", "brand_owner", "market_country", "description", "ingredients"
    ] + C.NUT11
    full = full[[c for c in ordered_cols if c in full.columns]]

    # Apply validity filter to produce non-NaN valid rows where possible
    try:
        full_valid = filter_valid_nutrition_data(full)
    except Exception:
        full_valid = full.copy()

    return full_valid


__all__ = [
    "compute_zero_thresholds",
    "build_snack_input_df_new",
    "make_full_df_nona",
    "prepare_until_full_df_nona",
    "filter_valid_nutrition_data",
    "prepare_pipeline",
    "embed_ingredients_and_save",
    "compute_umap_from_embeddings",
]
