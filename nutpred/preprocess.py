import ast
import re
import logging
from typing import List, Tuple, Optional
from collections import Counter

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)

# -----------------------------
# Ingredient string cleaning
# -----------------------------
def split_top_level(text: str) -> List[str]:
    """Split by top-level commas/periods (ignore those inside () or [])."""
    parts, buf = [], []
    depth_par = depth_brk = 0
    for c in text:
        if c == "(":
            depth_par += 1; buf.append(c)
        elif c == ")":
            depth_par = max(depth_par - 1, 0); buf.append(c)
        elif c == "[":
            depth_brk += 1; buf.append(c)
        elif c == "]":
            depth_brk = max(depth_brk - 1, 0); buf.append(c)
        elif (c in {",", "."}) and depth_par == 0 and depth_brk == 0:
            part = "".join(buf).strip()
            if part: parts.append(part)
            buf = []
        else:
            buf.append(c)
    last = "".join(buf).strip()
    if last: parts.append(last)
    return parts

_CONTAINS_PREFIX = re.compile(
    r"""(?ix) ^
        \s*
        (?:
            (?:and\s+)?less\s+than\s+\d+%?\s+of(?:\s+the\s+following)?\s*:? |
            (?:may\s+)?contains?\s+\d+%?\s*(?:or\s+less|or\s+more)?\s*(?:of\s*:?) |
            (?:may\s+)?contains?\s+(?:one\s+or\s+more\s+of\s+the\s+following:\s*) |
            (?:may\s+)?contains?\s*:?
        )
        \s*
    """
)

def _strip_contains_prefix(s: str) -> str:
    return _CONTAINS_PREFIX.sub("", s).strip()

def clean_ingredient_text(text: str) -> List[str]:
    """
    Turn an INGREDIENTS string into a clean list of tokens.
    - drops bracketed sublists but keeps the head phrase
    - strips 'contains/less than 2% of the following' style prefixes
    """
    text = str(text).strip()
    if not text:
        logger.debug("Empty text input for ingredient cleaning")
        return []
    
    logger.debug(f"Cleaning ingredient text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    
    text = text.lower()
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"^\s*ingredients?\s*:\s*", "", text, flags=re.I)
    parts = split_top_level(text)

    cleaned: List[str] = []
    for i, part in enumerate(parts):
        ing = part.strip()
        if not ing:
            continue

        logger.debug(f"Processing ingredient part {i+1}: '{ing}'")

        # explicit "(contains ...)" → drop
        original_ing = ing
        ing = re.sub(r"\(\s*(?:may\s+)?contains?.*?\)", "", ing, flags=re.I)
        if original_ing != ing:
            logger.debug(f"Removed contains clause: '{original_ing}' -> '{ing}'")

        # drop bracketed/parenthetical sublists; keep head
        original_ing = ing
        ing = re.sub(r"\[[^\]]*\]", "", ing)
        ing = re.sub(r"\([^)]*\)", "", ing)
        if original_ing != ing:
            logger.debug(f"Removed brackets/parentheses: '{original_ing}' -> '{ing}'")

        # strip leading "contains/less than 2% ..." prefixes
        ing = _strip_contains_prefix(ing)

        # leading connectors
        original_ing = ing
        ing = re.sub(r"^(?:and\/or|and or|and|or|andor)\s+", "", ing)
        if original_ing != ing:
            logger.debug(f"Removed leading connector: '{original_ing}' -> '{ing}'")

        # descriptive lead-ins
        original_ing = ing
        ing = re.sub(r"^\s*(?:made\s+with|made\s+of|made\s+from)\s*:\s*", "", ing)
        ing = re.sub(r"^\s*(?:includes?|including|containing)\s*:\s*", "", ing)
        ing = re.sub(r"^\s*if[^,;]*", "", ing)
        if original_ing != ing:
            logger.debug(f"Removed descriptive lead-in: '{original_ing}' -> '{ing}'")

        # fallback prefix
        original_ing = ing
        ing = re.sub(r"^\s*\d+%?\s*(?:or\s+less|or\s+more)?\s*(?:of\s*:?)\s*", "", ing)
        if original_ing != ing:
            logger.debug(f"Removed fallback prefix: '{original_ing}' -> '{ing}'")

        # normalize
        original_ing = ing
        ing = re.sub(r"[^a-z0-9\s\-&/#]", "", ing).strip()
        ing = re.sub(r"\s+", " ", ing)
        ing = ing.replace("andor", "and/or")
        if ing in ("natural flavors", "natural flavoring"): 
            ing = "natural flavor"
            logger.debug("Normalized 'natural flavors' -> 'natural flavor'")
        if ing == "thiamine mononitrate": 
            ing = "thiamin mononitrate"
            logger.debug("Normalized 'thiamine mononitrate' -> 'thiamin mononitrate'")
        ing = re.sub(r"(?:\s+(?:and|or|and\/or))+$", "", ing)
        
        if original_ing != ing:
            logger.debug(f"Normalized text: '{original_ing}' -> '{ing}'")

        if ing:
            cleaned.append(ing)
            logger.debug(f"Added cleaned ingredient: '{ing}'")
        else:
            logger.debug("Skipped empty ingredient after cleaning")

    logger.debug(f"Ingredient cleaning complete. Input: {len(parts)} parts, Output: {len(cleaned)} ingredients")
    return cleaned

# -----------------------------
# Public API
# -----------------------------
def load_inputs(snack_csv: str, thesaurus_xlsx: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load snack CSV and thesaurus XLSX."""
    logger.info(f"Loading snack data from: {snack_csv}")
    snack = pd.read_csv(snack_csv)
    logger.info(f"Loaded snack data: {snack.shape}")
    
    logger.info(f"Loading thesaurus from: {thesaurus_xlsx}")
    thesaurus = pd.read_excel(thesaurus_xlsx)
    logger.info(f"Loaded thesaurus: {thesaurus.shape}")
    
    return snack, thesaurus

def ensure_mapped_list_column(
    snack: pd.DataFrame,
    thesaurus: pd.DataFrame,
    ing_col_candidates=("ingredients", "ingredient_list"),
) -> pd.DataFrame:
    """
    Create snack['mapped_list'] using thesaurus mapping over parsed token lists.
    Accepts either a raw 'ingredients' string or a pre-existing 'ingredient_list'.
    """
    logger.info("Creating mapped_list column using thesaurus mapping")
    
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
            logger.info(f"Using thesaurus columns: '{a}' -> '{b}'")
            break
    if mapping is None:
        logger.error("Thesaurus must have 'Parsed ingredient term' and 'Preferred descriptor' columns.")
        raise ValueError("Thesaurus must have 'Parsed ingredient term' and 'Preferred descriptor' columns.")

    # locate raw ingredient column
    raw_col = next((c for c in ing_col_candidates if c in snack.columns), None)
    if raw_col is None:
        logger.error(f"Snack CSV must contain one of columns: {ing_col_candidates}")
        raise ValueError(f"Snack CSV must contain one of columns: {ing_col_candidates}")
    
    logger.info(f"Using ingredient column: {raw_col}")

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
            mapped = mapping.get(t, t)
            if mapped != t:
                logger.debug(f"Mapped ingredient: '{t}' -> '{mapped}'")
            out.append(mapped)
        return out

    snack = snack.copy()
    logger.info("Creating ingredient_list_raw column")
    snack["ingredient_list_raw"] = snack[raw_col].apply(to_list)
    
    logger.info("Creating mapped_list column")
    snack["mapped_list"] = snack["ingredient_list_raw"].apply(map_terms)
    
    # Log some statistics
    total_ingredients = sum(len(lst) for lst in snack["mapped_list"])
    unique_ingredients = len(set(ing for lst in snack["mapped_list"] for ing in lst))
    logger.info(f"Mapped {total_ingredients} total ingredients to {unique_ingredients} unique ingredients")
    
    return snack

def filter_rows(snack: pd.DataFrame, exclude_terms=("popcorn", "pretzel", "pretzels")) -> pd.DataFrame:
    """Drop rows whose mapped_list contains any of the excluded terms."""
    def has_excluded(lst):
        if not isinstance(lst, list):
            return False
        low = [str(x).lower() for x in lst]
        return any(any(term in s for term in exclude_terms) for s in low)
    mask = snack["mapped_list"].apply(has_excluded)
    return snack.loc[~mask].reset_index(drop=True)

def make_topk(snack: pd.DataFrame, k: int) -> List[str]:
    """Return the K most frequent mapped ingredients."""
    freq = Counter()
    for lst in snack["mapped_list"]:
        if isinstance(lst, list):
            freq.update(lst)
    return [ing for ing, _ in freq.most_common(k)]

def build_binary_and_scores(snack: pd.DataFrame, top_list: List[str], max_score: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (binary_df, score_df) over the given Top-K ingredient universe."""
    lists = snack["mapped_list"].apply(lambda x: x if isinstance(x, list) else [])
    # binary
    mlb = MultiLabelBinarizer(classes=top_list)
    bin_mat = mlb.fit_transform(lists)
    binary_df = pd.DataFrame(bin_mat, columns=[f"binary_{c}" for c in mlb.classes_], index=snack.index)
    # scored (position-based)
    def score_row(lst, top):
        sc = {ing: 0 for ing in top}
        for idx, ing in enumerate(lst[:max_score]):
            if ing in sc:
                sc[ing] = max_score - idx
        return [sc[i] for i in top]
    score_mat = [score_row(lst, top_list) for lst in lists]
    score_df = pd.DataFrame(score_mat, columns=[f"score_{c}" for c in top_list], index=snack.index)
    return binary_df, score_df

def ensure_umap_columns(snack: pd.DataFrame, expected_dim: int = 10) -> pd.DataFrame:
    """
    Use provided embeddings. If a single 'umap_10' column exists (list-like/string),
    expand it into 'umap_10_1'..'umap_10_10'. If already expanded, do nothing.
    """
    expanded = [c for c in snack.columns if c.startswith("umap_10_")]
    if expanded:
        return snack

    if "umap_10" in snack.columns:
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

        arrs = snack["umap_10"].apply(parse_cell).tolist()
        arrs = [a[:expected_dim] + [0.0] * max(0, expected_dim - len(a)) for a in arrs]
        for i in range(expected_dim):
            snack[f"umap_10_{i+1}"] = [a[i] for a in arrs]
    return snack

def select_base_nutrients(snack: pd.DataFrame) -> List[str]:
    cols = [
        "Energy(kcal)", "Carbohydrate(g)", "Total fat(g)", "Protein(g)",
        "Sodium(mg)", "Total sugar(g)", "Saturated fatty acids(g)", "Cholesterol(mg)",
    ]
    missing = [c for c in cols if c not in snack.columns]
    if missing:
        raise ValueError(f"Missing base nutrient columns: {missing}")
    return cols

def ensure_targets(snack: pd.DataFrame) -> List[str]:
    targets = ["Calcium(mg)", "Fiber(g)", "Iron(mg)"]
    missing = [c for c in targets if c not in snack.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
    return targets

__all__ = [
    "clean_ingredient_text",
    "load_inputs",
    "ensure_mapped_list_column",
    "filter_rows",
    "make_topk",
    "build_binary_and_scores",
    "ensure_umap_columns",
    "select_base_nutrients",
    "ensure_targets",
]
