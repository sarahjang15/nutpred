import ast
import re
import logging
from typing import List, Tuple, Optional
from collections import Counter
from nutpred.cleaning import clean_ingredient_text
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer 

logger = logging.getLogger(__name__)


# -----------------------------
# Public API
# -----------------------------
def load_inputs(food_df_csv: str, thesaurus_xlsx: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load food_df CSV and thesaurus XLSX."""
    food_df = pd.read_csv(food_df_csv)
    thesaurus = pd.read_excel(thesaurus_xlsx)
    return food_df, thesaurus

def make_id_col(food_df: pd.DataFrame) -> pd.DataFrame:
    """Make id column from index"""
    food_df = food_df.copy()
    food_df['id_col'] = food_df.index.astype(str)
    return food_df

def ensure_mapped_list_column(
    food_df: pd.DataFrame,
    thesaurus: pd.DataFrame,
    ing_col_candidates=("ingredients", "ingredient_list"),
) -> pd.DataFrame:
    """
    Create food_df['mapped_list'] using thesaurus mapping over parsed token lists.
    Accepts either a raw 'ingredients' string or a pre-existing 'ingredient_list'.
    """
    # thesaurus mapping (case-insensitive fallbacks)
    possible_t_cols = [
        ("Parsed ingredient term", "Preferred descriptor"),
        ("parsed ingredient term", "preferred descriptor"),
        ("PARSED INGREDIENT TERM", "PREFERRED DESCRIPTOR"),
    ]
    mapping = None
    for a, b in possible_t_cols:
        if a in thesaurus.columns and b in thesaurus.columns:
            mapping = thesaurus.set_index(a)[b].to_dict()
            break
    if mapping is None:
        raise ValueError("Thesaurus must have 'Parsed ingredient term' and 'Preferred descriptor' columns.")

    # locate raw ingredient column
    raw_col = next((c for c in ing_col_candidates if c in food_df.columns), None)
    if raw_col is None:
        raise ValueError(f"food_df CSV must contain one of columns: {ing_col_candidates}")

    def to_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            # allow list-like strings
            try:
                maybe = ast.literal_eval(x)
                if isinstance(maybe, list):
                    return maybe
            except Exception:
                pass
            return clean_ingredient_text(x)
        return []

    def map_terms(lst: List[str]) -> List[str]:
        out: List[str] = []
        for term in lst:
            t = str(term).strip().lower()
            out.append(mapping.get(t, t))
        return out

    food_df = food_df.copy()
    food_df["ingredient_list_raw"] = food_df[raw_col].apply(to_list)
    
    # Create ingredient_list_top20 (first 20 items from ingredient_list_raw)
    def take_top20(lst):
        if isinstance(lst, list):
            return lst[:20]
        return []
    
    food_df["ingredient_list_top20"] = food_df["ingredient_list_raw"].apply(take_top20)
    food_df["mapped_list"] = food_df["ingredient_list_top20"].apply(map_terms)
    
    # Log ingredient counts
    total_raw_ingredients = sum(len(lst) if isinstance(lst, list) else 0 for lst in food_df["ingredient_list_raw"])
    total_top20_ingredients = sum(len(lst) if isinstance(lst, list) else 0 for lst in food_df["ingredient_list_top20"])
    total_mapped_ingredients = sum(len(lst) if isinstance(lst, list) else 0 for lst in food_df["mapped_list"])
    
    logger.info(f"Ingredient processing: {total_raw_ingredients} raw ingredients -> {total_top20_ingredients} top20 ingredients -> {total_mapped_ingredients} mapped ingredients")
    
    return food_df

def filter_by_category(food_df: pd.DataFrame, categories: list) -> pd.DataFrame:
    """
    Filter food_df by branded_food_category.
    
    Args:
        food_df: DataFrame with all food categories
        categories: List of categories to filter by, or ["full"] for all data
    
    Returns:
        filtered_df: DataFrame with selected categories
    """
    if categories == ["full"]:
        logger.info("Using full dataset without category filtering")
        return food_df.copy()
    
    if "branded_food_category" not in food_df.columns:
        logger.warning("branded_food_category column not found, returning full dataset")
        return food_df.copy()
    
    # Convert categories to lowercase for case-insensitive matching
    categories_lower = [cat.lower() for cat in categories]
    
    # Filter rows where branded_food_category contains any of the specified categories
    pattern = '|'.join(categories_lower)
    mask = food_df['branded_food_category'].str.lower().str.contains(pattern, na=False)
    
    filtered_df = food_df[mask].copy()
    logger.info(f"Filtered from {len(food_df)} to {len(filtered_df)} samples using categories: {categories}")
    
    # Log some examples of what was found
    if len(filtered_df) > 0:
        sample_categories = filtered_df['branded_food_category'].head(3).tolist()
        logger.info(f"Sample filtered categories: {sample_categories}")
    
    return filtered_df

def filter_by_ingredients(food_df: pd.DataFrame, include_terms=("popcorn", "pretzel", "pretzels")) -> pd.DataFrame:
    """Keep only rows whose mapped_list contains any of the included terms."""
    def has_included(lst):
        if not isinstance(lst, list):
            return False
        low = [str(x).lower() for x in lst]
        return any(any(term in s for term in include_terms) for s in low)
    mask = food_df["mapped_list"].apply(has_included)
    return food_df.loc[mask].reset_index(drop=True)

def filter_rows(food_df: pd.DataFrame, exclude_terms=("popcorn", "pretzel", "pretzels")) -> pd.DataFrame:
    """Drop rows whose mapped_list contains any of the excluded terms."""
    def has_excluded(lst):
        if not isinstance(lst, list):
            return False
        low = [str(x).lower() for x in lst]
        return any(any(term in s for term in exclude_terms) for s in low)
    mask = food_df["mapped_list"].apply(has_excluded)
    return food_df.loc[~mask].reset_index(drop=True)

def make_topk(food_df: pd.DataFrame, k: int) -> List[str]:
    """Return the K most frequent mapped ingredients."""
    freq = Counter()
    for lst in food_df["mapped_list"]:
        if isinstance(lst, list):
            freq.update(lst)
    return [ing for ing, _ in freq.most_common(k)]

def extract_topk_from_ingnut(ingnut_df: pd.DataFrame, k: int = None) -> Tuple[pd.DataFrame, List[str]]:
    """Extract top-k ingredients from ingnut_df with deduplication, keeping first occurrence."""
    ingredient_col = "ing"
    
    if ingredient_col not in ingnut_df.columns:
        raise ValueError(f"Column '{ingredient_col}' not found in ingnut_df")
    
    full_ingredients = ingnut_df[ingredient_col].tolist()

    # Keep only the first occurrence of each ingredient
    filtered_ingnut_df = ingnut_df.drop_duplicates(subset=[ingredient_col], keep='first')
    unique_ingredients = filtered_ingnut_df[ingredient_col].tolist()
    logger.info(f"Extracted {len(unique_ingredients)} unique ingredients from ingnut_df (keeping first occurrence)")
    logger.info(f"Filtered ingnut_df from {len(ingnut_df)} to {len(filtered_ingnut_df)} rows")
    logger.info(f"Removed {len(ingnut_df) - len(filtered_ingnut_df)} duplicate ingredient rows")
    
    return filtered_ingnut_df, unique_ingredients  

def filter_ingredients_to_topk(food_df: pd.DataFrame, top_list: List[str]) -> pd.DataFrame:
    """
    Filter mapped_list to only include ingredients that are in the top-k list from ingnut_df.
    Returns a new column 'mapped_list_topk_only' with filtered ingredients.
    """
    logger.info("Filtering ingredients to top-k list...")
    
    def filter_to_topk(ingredient_list):
        if not isinstance(ingredient_list, list):
            return []
        
        filtered_list = []

        for ingredient in ingredient_list:
            # Check if the ingredient (or its base name) is in top_list
            if ingredient in top_list:
                filtered_list.append(ingredient)
        
        return filtered_list
    
    food_df = food_df.copy()
    food_df['mapped_list_topk_only'] = food_df['mapped_list'].apply(filter_to_topk)
    
    # Log statistics
    total_original = sum(len(lst) if isinstance(lst, list) else 0 for lst in food_df['mapped_list'])
    total_filtered = sum(len(lst) if isinstance(lst, list) else 0 for lst in food_df['mapped_list_topk_only'])
            
    logger.info(f"Filtered ingredients: {total_original} original -> {total_filtered} in top-k ({total_filtered/max(total_original, 1)*100:.1f}%)")
    
    return food_df

# Create first_mapped filter
def is_first_mapped(row):
    # Check if the first ingredient from ingredient_list_top20 is properly mapped and in top-k
    ingredient_list_top20 = row.get('ingredient_list_top20', [])
    mapped_list = row.get('mapped_list', [])
    mapped_topk_list = row.get('mapped_list_topk_only', [])
        
    if not isinstance(ingredient_list_top20, list) or len(ingredient_list_top20) == 0:
        return False
        
    if not isinstance(mapped_list, list) or len(mapped_list) == 0:
        return False
        
    if not isinstance(mapped_topk_list, list) or len(mapped_topk_list) == 0:
        return False
        
    # Get the first ingredient from the original top20 list
    first_raw_ing = ingredient_list_top20[0].strip()
    if first_raw_ing == "":
        return False
        
    # Get the first ingredient from the mapped list (this should be the mapped version of the first raw ingredient)
    first_mapped_ing = mapped_list[0].strip()
    if first_mapped_ing == "":
        return False

    # The first_mapped is TRUE if the first mapped ingredient is present in the top-k list
    # This means it was successfully mapped and survived the top-k filtering
    return first_mapped_ing in mapped_topk_list
    


def build_binary_and_scores(food_df: pd.DataFrame, top_list: List[str], max_score: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create binary and score features from mapped_list_topk_only (resolved ingredients)
    """
    logger.info("Creating binary and score features from ingredients...")
    
    # Use mapped_list_topk_only instead of mapped_list
    lists = food_df["mapped_list_topk_only"].apply(lambda x: x if isinstance(x, list) else [])
    
    mlb = MultiLabelBinarizer()

    bin_mat = mlb.fit_transform(lists)
    binary_df = pd.DataFrame(bin_mat, columns=[f"binary_{c}" for c in mlb.classes_], index=food_df.index)
    
    # Create score features using resolved ingredients (keep the variants for scoring)
    def score_row(lst, top):
        sc = {ing: 0 for ing in top}
        for idx, ing in enumerate(lst[:max_score]):
            if ing in sc:
                sc[ing] = max_score - idx
        return [sc[i] for i in top]
    
    score_mat = [score_row(lst, top_list) for lst in lists]
    score_df = pd.DataFrame(score_mat, columns=[f"score_{c}" for c in top_list], index=food_df.index)
    
    logger.info(f"Created {len(binary_df.columns)} binary features and {len(score_df.columns)} score features")
    return binary_df, score_df

def ensure_umap_columns(food_df: pd.DataFrame, expected_dim: int = 10) -> pd.DataFrame:
    """
    Use provided embeddings. If a single 'umap_10' column exists (list-like/string),
    expand it into 'umap_10_1'..'umap_10_10'. If already expanded, do nothing.
    """
    expanded = [c for c in food_df.columns if c.startswith("umap_10_")]
    if expanded:
        return food_df

    if "umap_10" in food_df.columns:
        def parse_cell(v):
            if isinstance(v, (list, tuple)):
                return list(v)
            if isinstance(v, str):
                try:
                    arr = ast.literal_eval(v)
                    if isinstance(arr, (list, tuple)):
                        return list(arr)
                except Exception:
                    pass
                v2 = v.strip().strip("[]")
                if not v2:
                    return [0.0] * expected_dim
                try:
                    return [float(x) for x in v2.split(",")]
                except Exception:
                    return [0.0] * expected_dim
            return [0.0] * expected_dim

        arrs = food_df["umap_10"].apply(parse_cell).tolist()
        arrs = [a[:expected_dim] + [0.0] * max(0, expected_dim - len(a)) for a in arrs]
        for i in range(expected_dim):
            food_df[f"umap_10_{i+1}"] = [a[i] for a in arrs]
    return food_df

def select_base_nutrients(food_df: pd.DataFrame) -> List[str]:
    cols = [
        "Energy(kcal)", "Carbohydrate(g)", "Total fat(g)", "Protein(g)",
        "Sodium(mg)", "Total sugar(g)", "Saturated fatty acids(g)", "Cholesterol(mg)",
    ]
    missing = [c for c in cols if c not in food_df.columns]
    if missing:
        raise ValueError(f"Missing base nutrient columns: {missing}")
    return cols

def ensure_targets(food_df: pd.DataFrame) -> List[str]:
    targets = ["Calcium(mg)", "Fiber(g)", "Iron(mg)"]
    missing = [c for c in targets if c not in food_df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
    return targets

 # Calculate mapped ratio
def calculate_mapped_ratio(row):
    raw_count = len(row.get('ingredient_list_raw', []))
    mapped_count = len(row.get('mapped_list_topk_only', []))
    return round(mapped_count / max(raw_count, 1), 4)

def calculate_mapped_ratio_top20(row):
    raw_count = len(row.get('ingredient_list_top20', []))
    mapped_count = len(row.get('mapped_list_topk_only', []))
    return round(mapped_count / max(raw_count, 1), 4)

def make_filters(food_df: pd.DataFrame) -> pd.DataFrame:

    first_mapped_mask = food_df.apply(is_first_mapped, axis=1)
    food_df["first_mapped"] = first_mapped_mask
    food_df["mapped_ratio"] = food_df.apply(calculate_mapped_ratio, axis=1)
    food_df["mapped_ratio_top20"] = food_df.apply(calculate_mapped_ratio_top20, axis=1)

    filter_combinations = [
    # Filter 1: first_mapped = TRUE
    {"name": "first_mapped", "condition": lambda df: df['first_mapped'] == True},
    
    # Filter 2: mapped_ratio_top20 >= 0.8
    {"name": "mapped_ratio_top20_high", "condition": lambda df: df['mapped_ratio_top20'] >= 0.8},
    
    # Filter 3: mapped_ratio >= 0.8
    {"name": "mapped_ratio_high", "condition": lambda df: df['mapped_ratio'] >= 0.8},

    # Filter 4: strict
    {"name": "strict", "condition": lambda df: df['first_mapped'] == True and df['mapped_ratio_top20'] >= 0.8}
    ]

    for filter in filter_combinations:
        food_df[filter['name']] = food_df.apply(filter['condition'], axis=1)
    return food_df
        
def preprocess_pipeline(food_df: pd.DataFrame, ingnut_df: pd.DataFrame, k: int = 133) -> pd.DataFrame:
    """
    Complete ingredient processing pipeline:
    0. Make id column
    1. Filter to top-k (mapped_list_topk_only)
    2. Create binary/score features from resolved ingredients
    3. Make filters
    4. Join features to food_df dataframe
    """
    logger.info("Preprocessing ingredients...")
    
    # Step 1: Extract top-k ingredients from ingnut_df
    ingnut_df, top_list = extract_topk_from_ingnut(ingnut_df, k=k)
    
    # Step 2: Filter to top-k 
    food_df = filter_ingredients_to_topk(food_df, top_list)
    
    # Step 3: Make filters
    food_df = make_filters(food_df)
    
    # Step 4: Create binary/score features from resolved ingredients
    binary_df, score_df = build_binary_and_scores(food_df, top_list)
    
    # Step 5: Join features to food_df dataframe
    food_df = food_df.join(binary_df).join(score_df)
    
    logger.info("Completed ingredient preprocessing.")
    return food_df
    
__all__ = [
    "load_inputs",
    "make_id_col",
    "ensure_mapped_list_column",
    "filter_by_category",
    "filter_by_ingredients",
    "filter_rows",
    "make_topk",
    "extract_topk_from_ingnut",
    "filter_ingredients_to_topk",
    "is_first_mapped",
    "make_filters",
    "process_ingredients",
    "build_binary_and_scores",
    "ensure_umap_columns",
    "select_base_nutrients",
    "ensure_targets",
    "calculate_mapped_ratio",
    "calculate_mapped_ratio_top20",
]
