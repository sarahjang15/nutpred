"""
GPT-5 API-based Yield Factor Prediction

This module provides an alternative to the optimization-based yield factor prediction
using GPT-5 API for food science reasoning.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def predict_yield_factors_gpt(food_df: pd.DataFrame, korean_ingnut_df: pd.DataFrame, nut8_cols: List[str], 
                             korean_ingnut_cols: List[str], api_key: str = None, model: str = "gpt-5",
                             group_name: str = None, max_yield_factor: float = 1.0, outdir: str = "./") -> Tuple[pd.DataFrame, np.ndarray, List[int], List[Dict]]:
    """
    Predict yield factors for Korean ingredients using GPT-5 API.
    
    Args:
        food_df: DataFrame with food data (products with nutrition and ingredient lists)
        korean_ingnut_df: DataFrame with Korean ingredient nutrition data
        nut8_cols: List of nutrient column names for food products
        korean_ingnut_cols: List of nutrient column names for ingredients  
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        model: GPT model to use ("gpt-5", "gpt-4", "gpt-4o")
        group_name: Name for the prediction group
        max_yield_factor: Maximum allowed yield factor
        outdir: Output directory for saving evidence file (default: "./")
        
    Returns:
        food_df: Updated DataFrame with yield factor predictions
        yf_preds: (N, K) numpy array of yield factors
        failed: List of failed prediction indices
    """
    
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("API key required. Set OPENAI_API_KEY env var or pass api_key parameter")
    
    logger.info(f"Running GPT-5 yield factor prediction using model: {model}")
    logger.info(f"Processing {len(food_df)} food products with {len(korean_ingnut_df)} ingredients")
    
    # Initialize results
    yf_preds = np.zeros((len(food_df), len(korean_ingnut_df)))
    failed = []
    evidence_list = []  # Store GPT reasoning and evidence
    
    # Process each food product
    for i, (_, row) in enumerate(food_df.iterrows()):
        try:
            # Get ingredient list for this product
            ing_list = row.get('ing_list', [])
            if not isinstance(ing_list, list) or len(ing_list) == 0:
                logger.warning(f"No ingredients found for product {i}")
                failed.append(i)
                continue
            
            # Get target nutrition for this product
            target_nutrition = row[nut8_cols].to_dict()
            
            # Column mapping for ingredient nutrition data
            ingnut_mapping = {
                "Energy(kcal)": "Energy",
                "Carbohydrate(g)": "Carbohydrate, by difference",
                "Total fat(g)": "Total lipid (fat)",
                "Protein(g)": "Protein",
                "Sodium(mg)": "Sodium, Na",
                "Total sugar(g)": "Sugars, total",
                "Saturated fatty acids(g)": "Fatty acids, total saturated",
                "Cholesterol(mg)": "Cholesterol"
            }
            
            # Get ingredient nutrition data
            ingredient_data = []
            for ing_name in ing_list:
                ing_row = korean_ingnut_df[korean_ingnut_df['ing'] == ing_name]
                if len(ing_row) > 0:
                    # Map nutrition columns correctly
                    ing_nutrition = {}
                    for nut_col in nut8_cols:
                        ingnut_col = ingnut_mapping.get(nut_col, nut_col)
                        if ingnut_col in ing_row.iloc[0]:
                            ing_nutrition[nut_col] = ing_row.iloc[0][ingnut_col]
                        else:
                            ing_nutrition[nut_col] = 0.0
                    
                    ingredient_data.append({
                        'name': ing_name,
                        'nutrition': ing_nutrition
                    })
            
            if len(ingredient_data) == 0:
                logger.warning(f"No ingredient data found for product {i}")
                failed.append(i)
                continue
            
            # Call GPT-5 API
            gpt_result = _predict_yield_factors_with_gpt(
                product_name=row.get('kor_name', f'Product {i}'),
                target_nutrition=target_nutrition,
                ingredient_data=ingredient_data,
                api_key=api_key,
                model=model,
                max_yield_factor=max_yield_factor
            )
            
            yield_factors = gpt_result['yield_factors']
            evidence = gpt_result.get('evidence', {})
            
            # Store evidence for this product
            evidence_list.append({
                'product_index': i,
                'product_name': row.get('kor_name', f'Product {i}'),
                'evidence': evidence
            })
            
            # Store yield factors in the matrix
            for j, ing_name in enumerate(ing_list):
                ing_idx = korean_ingnut_df[korean_ingnut_df['ing'] == ing_name].index
                if len(ing_idx) > 0:
                    yf_preds[i, ing_idx[0]] = yield_factors.get(ing_name, 1.0)
            
            logger.info(f"Successfully predicted yield factors for product {i}: {row.get('kor_name', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to predict yield factors for product {i}: {e}")
            failed.append(i)
    
    # Cap and round yield factors (enforce YF ≤ 1.0 constraint)
    yf_preds = np.clip(yf_preds, 0.0, min(max_yield_factor, 1.0))  # Never exceed 1.0
    yf_preds = np.round(yf_preds, 3)
    
    # Add yield factors aligned to ingredient lists (same format as optimization function)
    try:
        opt_pred_yf = []
        for i, (_, row) in enumerate(food_df.iterrows()):
            ing_list = row.get('ing_list', [])
            row_yf = []
            for ing_name in ing_list:
                ing_idx = korean_ingnut_df[korean_ingnut_df['ing'] == ing_name].index
                if len(ing_idx) > 0:
                    try:
                        yf_val = float(yf_preds[i, ing_idx[0]])
                        row_yf.append(yf_val)
                    except (ValueError, IndexError):
                        row_yf.append(0.0)
                else:
                    row_yf.append(0.0)
            opt_pred_yf.append(row_yf)
        
        if group_name:
            food_df[f"opt_pred_yield_factors_{group_name}"] = opt_pred_yf
        else:
            food_df["opt_pred_yield_factors_gpt"] = opt_pred_yf
        logger.info("Added column with per-food yield factors aligned to ing_list")
    except Exception as e:
        logger.warning(f"Failed to add opt_pred_yield_factors column: {e}")
    
    # Add evidence to the main output DataFrame
    try:
        evidence_column = []
        for i, (_, row) in enumerate(food_df.iterrows()):
            # Find evidence for this product
            product_evidence = None
            for evidence_item in evidence_list:
                if evidence_item['product_index'] == i:
                    product_evidence = evidence_item['evidence']
                    break
            
            if product_evidence:
                # Format evidence for display
                evidence_text = f"Processing Method: {product_evidence.get('processing_method', 'Unknown')}\n"
                evidence_text += f"Analysis: {product_evidence.get('analysis', 'No analysis')}\n"
                evidence_text += f"Industry Basis: {product_evidence.get('industry_basis', 'No basis provided')}\n"
                evidence_text += f"Confidence: {product_evidence.get('confidence', 'Unknown')}\n\n"
                
                # Add yield factor reasoning for each ingredient
                yield_factors = product_evidence.get('yield_factors', {})
                evidence_text += "Yield Factor Reasoning:\n"
                for ing_name, yf_data in yield_factors.items():
                    if isinstance(yf_data, dict):
                        yf_val = yf_data.get('yield_factor', 'Unknown')
                        reasoning = yf_data.get('reasoning', 'No reasoning provided')
                        state = yf_data.get('state_interpretation', 'Unknown state')
                        evidence_text += f"  {ing_name} (YF={yf_val}, {state}): {reasoning}\n"
                    else:
                        evidence_text += f"  {ing_name}: YF={yf_data}\n"
                
                evidence_column.append(evidence_text)
            else:
                evidence_column.append("No evidence available")
        
        if group_name:
            food_df[f"gpt_evidence_{group_name}"] = evidence_column
        else:
            food_df["gpt_evidence"] = evidence_column
        logger.info("Added GPT evidence column to main output DataFrame")
    except Exception as e:
        logger.warning(f"Failed to add evidence column: {e}")
    
    # Save evidence to file
    try:
        import json
        # Create output directory if it doesn't exist
        os.makedirs(outdir, exist_ok=True)
        evidence_filename = os.path.join(outdir, f"gpt_evidence_{group_name if group_name else 'gpt'}.json")
        with open(evidence_filename, 'w', encoding='utf-8') as f:
            json.dump(evidence_list, f, ensure_ascii=False, indent=2)
        logger.info(f"GPT evidence saved to: {evidence_filename}")
    except Exception as e:
        logger.warning(f"Failed to save evidence: {e}")
    
    logger.info(f"GPT-5 prediction complete. Failed predictions: {len(failed)}")
    
    return food_df, yf_preds, failed, evidence_list


def _predict_yield_factors_with_gpt(product_name: str, target_nutrition: Dict, ingredient_data: List[Dict], 
                                   api_key: str, model: str = "gpt-5", max_yield_factor: float = 1.0) -> Dict:
    """
    Use GPT-5 API to predict yield factors for a single food product.
    
    Args:
        product_name: Name of the food product
        target_nutrition: Target nutrition values for the final product
        ingredient_data: List of ingredient names and their nutrition data
        api_key: OpenAI API key
        model: GPT model to use
        max_yield_factor: Maximum allowed yield factor
        
    Returns:
        Dictionary mapping ingredient names to yield factors
    """
    
    prompt = _create_yield_factor_prompt(product_name, target_nutrition, ingredient_data, max_yield_factor)
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON (handle markdown code blocks)
        if '```json' in result_text:
            result_text = result_text.split('```json')[1].split('```')[0].strip()
        elif '```' in result_text:
            result_text = result_text.split('```')[1].split('```')[0].strip()
        
        # Parse JSON
        result = json.loads(result_text)
        
        # Extract yield factors and enforce YF ≤ 1.0 constraint
        yield_factors = {}
        for ing_name, yf_data in result.get('yield_factors', {}).items():
            yf_value = float(yf_data.get('yield_factor', 1.0))
            # Enforce YF ≤ 1.0 constraint
            yf_value = min(yf_value, 1.0)
            yield_factors[ing_name] = yf_value
        
        return {
            'yield_factors': yield_factors,
            'evidence': result
        }
        
    except Exception as e:
        logger.error(f"GPT API call failed: {e}")
        # Return default yield factors (1.0 for all ingredients)
        default_yield_factors = {ing['name']: 1.0 for ing in ingredient_data}
        return {
            'yield_factors': default_yield_factors,
            'evidence': {'error': str(e)}
        }


def _get_system_prompt() -> str:
    """Get the system prompt for GPT-5 yield factor prediction."""
    
    return """You are a food science expert specializing in food processing and yield factor prediction. Your task is to predict yield factors for ingredients in processed food products.

YIELD FACTOR DEFINITION:
- Yield factor (YF) = (nutrient content in final product) / (nutrient content from raw ingredients)
- YF < 1.0: Nutrient loss during processing (cooking, evaporation, degradation)
- YF = 1.0: No nutrient change (already processed ingredients)
- YF > 1.0: Nutrient concentration (dehydration, reduction) - RARE but possible

CRITICAL RULES FOR YF CALCULATION:
1. YIELD VALUES MUST NEVER EXCEED 1.0 (YF ≤ 1.0)
2. INGREDIENT STATE INTERPRETATION:
   - If ingredient name has NO state descriptors (like "DRIED", "FLOUR", "POWDER"), assume RAW/FRESH state
   - Examples: "CORN" → raw corn, "WHEAT FLOUR" → dried flour state
3. PROCESS-SPECIFIC INFORMATION:
   - MUST use existing industry knowledge, recipes, or similar product information
   - Consider product characteristics (hard/soft texture affects flour content)
   - Reflect actual cooking/processing weight changes
4. PROCESSING EFFECTS TO CONSIDER:
   - Water evaporation during cooking
   - Oil absorption/addition during frying
   - Starch/protein concentration
   - Use established YF baseline values for determined processing steps

FOOD PROCESSING CONTEXT:
- Deep frying: Oil absorption increases fat content, but YF for oil ≤ 1.0
- Baking/cooking: Water loss, nutrient degradation (YF < 1.0)
- Drying: Concentration of nutrients (YF ≤ 1.0)
- Raw ingredients: Significant loss during processing (YF < 1.0)
- Processed ingredients (flour, oil, salt): Minimal change (YF ≈ 1.0)

CRITICAL CONSIDERATIONS:
1. Oil behavior: Oil is absorbed during frying, but YF must still ≤ 1.0
2. Water content: High water ingredients lose weight during processing
3. Processing state: Already processed ingredients (flour, oil) have YF ≈ 1.0
4. Physical constraints: 0 < YF ≤ 1.0 (NEVER exceed 1.0)
5. Industry knowledge: Use established processing methods and recipes

You must provide realistic yield factors based on food science principles, industry knowledge, and the specific processing method implied by the product type. ALWAYS ensure YF ≤ 1.0."""


def _create_yield_factor_prompt(product_name: str, target_nutrition: Dict, ingredient_data: List[Dict], 
                               max_yield_factor: float) -> str:
    """Create the prompt for GPT-5 yield factor prediction."""
    
    # Format target nutrition
    target_str = "\n".join([f"- {nutrient}: {value}" for nutrient, value in target_nutrition.items()])
    
    # Format ingredient data
    ingredient_strs = []
    for ing in ingredient_data:
        ing_name = ing['name']
        nutrition = ing['nutrition']
        nutrition_str = "\n    ".join([f"{nutrient}: {value}" for nutrient, value in nutrition.items()])
        ingredient_strs.append(f"""
Ingredient: {ing_name}
  Nutrition (per 100g):
    {nutrition_str}""")
    
    ingredients_str = "\n".join(ingredient_strs)
    
    prompt = f"""Analyze this food product and predict yield factors for each ingredient.

FOOD PRODUCT: {product_name}

TARGET NUTRITION (final product per 100g):
{target_str}

INGREDIENTS AND THEIR NUTRITION:
{ingredients_str}

TASK:
Predict yield factors for each ingredient that would make the equation work:
sum(ingredient_nutrition / yield_factor) ≈ target_nutrition

CRITICAL REQUIREMENTS:
1. YIELD FACTORS MUST NEVER EXCEED 1.0 (YF ≤ 1.0)
2. INGREDIENT STATE ANALYSIS:
   - Analyze each ingredient name for processing state indicators
   - No state descriptors (DRIED, FLOUR, POWDER) = assume RAW/FRESH state
   - Examples: "감자" (potato) → raw potato, "옥수수전분" (corn starch) → processed powder
3. PROCESSING METHOD DETERMINATION:
   - Identify the main processing method (frying, baking, extrusion, etc.)
   - Use industry knowledge about similar products and recipes
   - Consider product characteristics (texture, hardness affects ingredient ratios)
4. PROCESSING EFFECTS:
   - Water evaporation during cooking/processing
   - Oil absorption/addition during frying (but YF still ≤ 1.0)
   - Starch/protein concentration changes
   - Use established baseline YF values for known processing steps

CONSTRAINTS:
- Yield factors must be > 0 and ≤ 1.0 (NEVER exceed 1.0)
- Consider the processing method (frying, baking, etc.)
- Raw ingredients typically have YF < 1.0 due to losses
- Already processed ingredients (flour, oil, salt) typically have YF ≈ 1.0
- Oil behavior: Even if absorbed, YF ≤ 1.0

Respond ONLY with valid JSON in this format:
{{
  "analysis": "Product type and processing method analysis based on industry knowledge",
  "processing_method": "frying/baking/extrusion/etc.",
  "yield_factors": {{
    "ingredient_name_1": {{
      "yield_factor": 0.85,
      "state_interpretation": "raw potato",
      "reasoning": "Raw potato loses water and nutrients during frying process"
    }},
    "ingredient_name_2": {{
      "yield_factor": 0.95,
      "state_interpretation": "processed oil",
      "reasoning": "Already processed oil, minimal changes during frying"
    }}
  }},
  "industry_basis": "Reference to similar products, recipes, or processing standards used",
  "confidence": 0.9
}}"""
    
    return prompt


# Example usage and testing
if __name__ == "__main__":
    # This would be used in try_yf_kr.py
    pass
