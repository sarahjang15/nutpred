#!/usr/bin/env python3
import argparse
import csv
import re
from typing import Tuple

import pandas as pd


def _norm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.lower()).strip()


def infer_processing_method(description: str, ingredients: str) -> str:
    d = _norm_text(description)
    ing = _norm_text(ingredients)
    txt = f"{d} {ing}"

    # Popcorn first
    if ("popcorn" in txt) or ("popping corn" in txt) or ("popping" in txt and "corn" in txt):
        # Coated/drizzled/sweet kettle styles
        if any(k in txt for k in ["kettle corn", "drizzle", "drizzled", "coated", "marshmallow", "cocoa", "chocolate", "caramel", "cinnabon", "sweet & salty", "sweet and salty", "brown sugar"]) or _has_sweet_profile(txt):
            return "popped+coated"
        return "popped"

    # Fruit snacks / gummies (gelled)
    if any(k in txt for k in ["fruit snacks", "gushers", "gummy", "gelatin", "pectin", "agar-agar", "agar", "carnauba wax"]) and any(k in txt for k in ["corn syrup", "sugar", "juice", "concentrate", "modified corn starch", "modified starch", "pectin", "gelatin"]):
        return "gelled"

    # Bakery-like: pretzels, crackers, grahams
    if any(k in txt for k in ["pretzel", "cracker", "graham", "biscuit", "cookie"]) or ("enriched flour" in txt or "wheat flour" in txt):
        # Coated variants (e.g., muddy buddies)
        if any(k in txt for k in ["muddy buddies", "coated", "drizzled", "chocolate", "yogurt coating", "creme coating", "frosted", "icing"]):
            return "baked+coated"
        return "baked"

    # Potato/corn snacks fried (chips, crisps, fries, rings)
    if any(k in txt for k in ["chip", "chips", "crisps", "pringles", "fries", "onion rings", "beer battered"]):
        # Distinguish fried extruded fries/rings vs sliced chips
        if any(k in txt for k in ["fries", "onion rings", "beer battered"]) or ("degermed yellow cornmeal" in txt and "dried potatoes" in txt):
            return "fried"
        # Potato chips
        if ("potato" in txt or "potatoes" in txt) and any(k in txt for k in ["oil", "vegetable oil", "canola", "sunflower", "safflower", "cottonseed", "palm"]):
            return "fried"

    # Cereal mix coated (Chex muddy buddies)
    if any(k in txt for k in ["muddy buddies", "checks mix", "chex mix"]) and any(k in txt for k in ["coated", "chocolate", "palm kernel oil", "cocoa"]):
        return "baked+coated"

    # Fallback baked if flours dominant
    if any(k in txt for k in ["flour", "baked", "oven"]):
        return "baked"

    return "unknown"


def _has_sweet_profile(txt: str) -> bool:
    return any(k in txt for k in ["sugar", "cane sugar", "corn syrup", "fructose", "caramel", "chocolate", "brown sugar", "marshmallow"])  


def estimate_yield_and_base(method: str, fat_g: float, sugar_g: float, carbs_g: float) -> Tuple[float, float]:
    """
    Returns (pred_yield, pred_base_fraction_est)
    All inputs are per 100 g product.
    """
    fat_f = max(0.0, min(1.0, (fat_g or 0.0) / 100.0))
    sugar_f = max(0.0, min(1.0, (sugar_g or 0.0) / 100.0))
    carbs_f = max(0.0, min(1.0, (carbs_g or 0.0) / 100.0))

    def clip(x, lo, hi):
        return max(lo, min(hi, x))

    # Defaults
    if method == "fried":
        # Distinguish extruded vs potato chips via fat and sugar proxies
        # Potato chips: high fat (>= 25 g/100g), very low sugar
        if fat_g >= 25 and sugar_g <= 5:
            base = clip(1.0 - fat_f - 0.03, 0.50, 0.75)
            yield_factor = 0.35
        else:
            # Extruded fries/rings common yield ~0.65-0.67
            base = clip(1.0 - fat_f - sugar_f - 0.03, 0.55, 0.75)
            yield_factor = 0.66
        return yield_factor, base

    if method == "popped":
        base = clip(1.0 - fat_f - 0.03, 0.55, 0.75)
        return base, base

    if method == "popped+coated":
        base = clip(1.0 - fat_f - sugar_f - 0.04, 0.25, 0.45)
        return base, base

    if method == "baked":
        base = clip(1.0 - fat_f - 0.03, 0.60, 0.90)
        return base, base

    if method == "baked+coated":
        base = clip(1.0 - fat_f - sugar_f - 0.04, 0.35, 0.85)
        return base, base

    if method == "gelled":
        # Two regimes: higher-sugar (gushers-like) vs typical fruit snacks
        if sugar_g >= 45:
            return 0.493, 0.493
        return 0.561, 0.561

    # unknown: heuristic from composition
    base = clip(1.0 - fat_f - sugar_f - 0.05, 0.40, 0.80)
    return base, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Ensure required nutrient columns exist
    for c in ["Total fat(g)", "Total sugar(g)", "Carbohydrate(g)", "ingredients", "description"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    methods = []
    yields = []
    bases = []

    for _, row in df.iterrows():
        method = infer_processing_method(row.get("description", ""), row.get("ingredients", ""))
        yld, base = estimate_yield_and_base(
            method,
            float(row.get("Total fat(g)", 0) or 0),
            float(row.get("Total sugar(g)", 0) or 0),
            float(row.get("Carbohydrate(g)", 0) or 0),
        )
        methods.append(method)
        yields.append(yld)
        bases.append(base)

    df["pred_processing_method"] = methods
    df["pred_yield"] = yields
    df["pred_base_fraction_est"] = bases

    # Write with CSV default quoting minimal
    df.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)


if __name__ == "__main__":
    main()
