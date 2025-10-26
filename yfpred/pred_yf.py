#!/usr/bin/env python3
"""
Test script for yield factor prediction verification.
Tests if sum(ing_nut / yf) = food_nut equation works correctly.
"""

import pandas as pd
import numpy as np
import ast
import os
from pred_yf_kr import predict_yield_factors
from pred_yf_kr_gpt import predict_yield_factors_gpt
import nutpred.constants as C

def create_perfect_test_case(food_df, ingnut_df, nut_cols):
    """
    Create a perfect test case where sum(ing_nut / yf) = food_nut should work exactly.
    This creates simple synthetic data with known yield factors.
    """
    # Column mapping for nutrients
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
    
    # Create simple test case with 3 food products and 3 ingredients
    # Each food has exactly 2 ingredients with known yield factors
    
    # Simple food data
    test_food_df = pd.DataFrame({
        'no': [1, 2, 3],
        'kor_name': ['Test Food 1', 'Test Food 2', 'Test Food 3'],
        'Energy(kcal)': [100.0, 200.0, 150.0],
        'Carbohydrate(g)': [20.0, 40.0, 30.0],
        'Total fat(g)': [5.0, 10.0, 7.5],
        'Protein(g)': [10.0, 20.0, 15.0],
        'Sodium(mg)': [50.0, 100.0, 75.0],
        'Total sugar(g)': [2.0, 4.0, 3.0],
        'Saturated fatty acids(g)': [1.0, 2.0, 1.5],
        'Cholesterol(mg)': [0.0, 0.0, 0.0]
    })
    
    # Simple ingredient data with known yield factors
    # For each food, ingredients are designed so that sum(ing_nut / yf) = food_nut
    test_ingnut_df = pd.DataFrame({
        'no': [1, 1, 2, 2, 3, 3],
        'kor_name': ['Test Food 1', 'Test Food 1', 'Test Food 2', 'Test Food 2', 'Test Food 3', 'Test Food 3'],
        'ing': ['Ingredient_A', 'Ingredient_B', 'Ingredient_C', 'Ingredient_D', 'Ingredient_E', 'Ingredient_F'],
        'ing_name': ['Test Ing A', 'Test Ing B', 'Test Ing C', 'Test Ing D', 'Test Ing E', 'Test Ing F'],
        'Energy': [50.0, 50.0, 100.0, 100.0, 75.0, 75.0],  # Sum = 100, 200, 150
        'Carbohydrate, by difference': [10.0, 10.0, 20.0, 20.0, 15.0, 15.0],  # Sum = 20, 40, 30
        'Total lipid (fat)': [2.5, 2.5, 5.0, 5.0, 3.75, 3.75],  # Sum = 5, 10, 7.5
        'Protein': [5.0, 5.0, 10.0, 10.0, 7.5, 7.5],  # Sum = 10, 20, 15
        'Sodium, Na': [25.0, 25.0, 50.0, 50.0, 37.5, 37.5],  # Sum = 50, 100, 75
        'Sugars, total': [1.0, 1.0, 2.0, 2.0, 1.5, 1.5],  # Sum = 2, 4, 3
        'Fatty acids, total saturated': [0.5, 0.5, 1.0, 1.0, 0.75, 0.75],  # Sum = 1, 2, 1.5
        'Cholesterol': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Sum = 0, 0, 0
    })
    
    # The perfect yield factors are all 1.0 (no loss during processing)
    # So sum(ing_nut / 1.0) = sum(ing_nut) = food_nut
    
    return test_food_df, test_ingnut_df

def test_yield_factor_equation(nut_cols=None, scale="none", ridge=0.01, robust=False, 
                               solver_name="osqp", resolver="rule", test_case="real", error_type="l2", 
                               max_yield_factor=1.0, method="optimization", api_key=None, gpt_model="gpt-5",
                               outdir="./", food_df_path="potato_example_foodnut.csv", 
                               ingnut_df_path="potato_example_ingnut.csv", output_excel_name="potato_example_output.xlsx",
                               output_matrix_name="yield_factors_matrix.npy"):
    """
    Test if the yield factor equation sum(ing_nut / yf) = food_nut works.
    
    Args:
        nut_cols: List of nutrient columns to use (default: C.NUT8)
        scale: Scaling method ("none", "std", "pminmax", "logiqr")
        ridge: Ridge regularization parameter
        robust: Use robust loss function
        solver_name: Optimization solver name
        resolver: Variant resolution method
        test_case: "real" for real data, "perfect" for perfect test case
        method: "optimization" or "gpt" (default: optimization)
        api_key: OpenAI API key (required for GPT method)
        gpt_model: GPT model to use ("gpt-5", "gpt-4", "gpt-4o")
        outdir: Output directory for saving results (default: "./")
        food_df_path: Path to food DataFrame CSV file
        ingnut_df_path: Path to ingredient-nutrient DataFrame CSV file
        output_excel_name: Name of output Excel file
        output_matrix_name: Name of output numpy matrix file
    """
    
    print("="*80)
    print(f"Testing Yield Factor Equation: sum(ing_nut / yf) = food_nut")
    print(f"Method: {method}")
    if method == "optimization":
        print(f"Parameters: scale={scale}, ridge={ridge}, robust={robust}, solver={solver_name}")
    else:
        print(f"Parameters: model={gpt_model}, max_yield_factor={max_yield_factor}")
    print(f"Test case: {test_case}")
    print("="*80)
    
    if nut_cols is None:
        nut_cols = C.NUT8
    
    # Load data
    print("Loading data...")
    food_df = pd.read_csv(food_df_path)
    ingnut_df = pd.read_csv(ingnut_df_path)

    if test_case == "perfect":
        # Create a perfect test case where the equation should work exactly
        print("Creating perfect test case...")
        food_df, ingnut_df = create_perfect_test_case(food_df, ingnut_df, nut_cols)
    
    # Add ingredient lists
    if test_case == "perfect":
        food_df['ing_list'] = [
            ['Ingredient_A', 'Ingredient_B'],  # for Test Food 1
            ['Ingredient_C', 'Ingredient_D'],  # for Test Food 2
            ['Ingredient_E', 'Ingredient_F']   # for Test Food 3
        ]
    else:
        food_df['ing_list'] = [
    ['감자', '정제팜유', '정제소금'],  # for product 1
    ['감자', '치즈', '옥수수전분', '해바라기유', '버터', '기타소금'],  # for product 3
    ['감자', '치즈', '옥수수전분', '카레분말', '해바라기유', '버터', '기타소금'],  # for product 4
    ['감자', '해바라기유', '옥수수전분', '기타소금', '옥수수', '건파슬리', '양파플레이크'],  # for product 5
    ['감자', '식용유지류', '설탕', '정제소금', '토마토분말', '천연향료 (Pepper, black)', '간장분말', '체다치즈분말', '바질분말'],  # for product 6
    ['감자', '감자전분 [감자전분 [감자], 변성전분]', '주정', '양파', '정제소금'],  # for product 7
    ['감자', '식용유지류', '정제소금', '설탕단풍수액']  # for product 8
    ]

    print(f"Loaded {len(food_df)} food products and {len(ingnut_df)} ingredient entries")
    print(f"Using nutrients: {nut_cols}")
    
    # Run yield factor prediction
    print(f"\nRunning yield factor prediction using {method} method...")
    
    if method == "gpt":
        if api_key is None:
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("API key required for GPT method. Set OPENAI_API_KEY env var or pass api_key parameter")
        
        food_df_with_predictions, yf_preds, failed, evidence_list = predict_yield_factors_gpt(
            food_df=food_df,
            korean_ingnut_df=ingnut_df,
            nut8_cols=nut_cols,
            korean_ingnut_cols=nut_cols,
            api_key=api_key,
            model=gpt_model,
            group_name="korean_yf",
            max_yield_factor=max_yield_factor,
            outdir=outdir
        )
    else:
        # Use optimization method
        food_df_with_predictions, yf_preds, failed = predict_yield_factors(
            food_df=food_df,
            korean_ingnut_df=ingnut_df,
            nut8_cols=nut_cols,
            korean_ingnut_cols=nut_cols,
            resolver=resolver,
            ridge=ridge,
            robust=robust,
            solver_name=solver_name,
            scale=scale,
            group_name="korean_yf",
            error_type=error_type,
            max_yield_factor=max_yield_factor
        )

    print(f"Prediction complete. Failed optimizations: {len(failed)}")
    print(f"Yield factor matrix shape: {yf_preds.shape}")
    
    # Test the equation for each food product
    print("\n" + "="*80)
    print("Testing yield factor equation for each food product:")
    print("="*80)
    
    # Column mapping for nutrients
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
    
    total_errors = []
    
    for i, (_, food_row) in enumerate(food_df_with_predictions.iterrows()):
        food_name = food_row['kor_name']
        print(f"\nProduct {i+1}: {food_name}")
        print("-" * 60)
        
        # Get ingredient list and yield factors for this product
        ing_list = food_row['ing_list']
        yf_list = food_row['opt_pred_yield_factors_korean_yf']
        
        print(f"Ingredients: {ing_list}")
        print(f"Yield factors: {[f'{yf:.3f}' for yf in yf_list]}")
        
        # Test equation for each nutrient
        nutrient_errors = []
        
        for nut_col in C.NUT8:
            ingnut_col = ingnut_mapping[nut_col]
            
            # Get food nutrient value (target)
            food_nut_val = food_row[nut_col]
            
            # Calculate sum(ing_nut / yf) for this nutrient
            predicted_sum = 0.0
            
            for j, ing_name in enumerate(ing_list):
                # Find ingredient in ingnut_df
                ing_row = ingnut_df[ingnut_df['ing'] == ing_name]
                if len(ing_row) > 0:
                    # Get the specific row for this food product
                    ing_row = ing_row[ing_row['kor_name'] == food_name]
                    if len(ing_row) > 0:
                        ing_nut_val = ing_row.iloc[0][ingnut_col]
                        yf_val = yf_list[j]
                        
                        # Avoid division by zero
                        if yf_val > 0:
                            predicted_sum += ing_nut_val / yf_val
                        else:
                            predicted_sum += 0.0
                    else:
                        print(f"  Warning: No ingredient data found for {ing_name} in {food_name}")
                else:
                    print(f"  Warning: Ingredient {ing_name} not found in database")
            
            # Calculate error
            error = abs(food_nut_val - predicted_sum)
            relative_error = error / food_nut_val * 100 if food_nut_val > 0 else 0
            
            nutrient_errors.append(error)
            
            print(f"  {nut_col}:")
            print(f"    Food value: {food_nut_val:.2f}")
            print(f"    Predicted:  {predicted_sum:.2f}")
            print(f"    Error:      {error:.2f} ({relative_error:.1f}%)")
        
        avg_error = np.mean(nutrient_errors)
        total_errors.append(avg_error)
        print(f"  Average error: {avg_error:.2f}")
    
    # Overall results
    print("\n" + "="*80)
    print("OVERALL RESULTS:")
    print("="*80)
    print(f"Average error across all products and nutrients: {np.mean(total_errors):.2f}")
    print(f"Maximum error: {np.max(total_errors):.2f}")
    print(f"Minimum error: {np.min(total_errors):.2f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    
    # Save results
    output_excel_path = os.path.join(outdir, output_excel_name)
    food_df_with_predictions.to_excel(output_excel_path, index=False)
    print(f"\nResults saved to: {output_excel_path}")
    
    # Show evidence column info if using GPT method
    if method == "gpt":
        evidence_col = "gpt_evidence"
        if evidence_col in food_df_with_predictions.columns:
            print(f"GPT evidence saved in column: {evidence_col}")
            print("Evidence includes: processing method, analysis, industry basis, confidence, and yield factor reasoning")
    
    # Save yield factor matrix
    yf_matrix_path = os.path.join(outdir, output_matrix_name)
    np.save(yf_matrix_path, yf_preds)
    print(f"Yield factor matrix saved to: {yf_matrix_path}")
    
    return food_df_with_predictions, yf_preds

def run_test_suite(outdir="./", food_df_path="potato_example_foodnut.csv", 
                   ingnut_df_path="potato_example_ingnut.csv", output_excel_name="potato_example_output.xlsx",
                   output_matrix_name="yield_factors_matrix.npy"):
    """Run various test cases with different parameters."""
    
    print("="*80)
    print("YIELD FACTOR TEST SUITE")
    print("="*80)
    
    # Test 1: Perfect test case (should have zero error)
    print("\n" + "="*60)
    print("TEST 1: Perfect Test Case (should have zero error)")
    print("="*60)
    test_yield_factor_equation(test_case="perfect", scale="none", ridge=0.0, outdir=outdir,
                               food_df_path=food_df_path, ingnut_df_path=ingnut_df_path,
                               output_excel_name=output_excel_name, output_matrix_name=output_matrix_name)
    
    # Test 2: Real data with different scaling methods
    print("\n" + "="*60)
    print("TEST 2: Real Data - Different Scaling Methods")
    print("="*60)
    
    for scale in ["none", "std", "pminmax"]:
        print(f"\n--- Scale: {scale} ---")
        test_yield_factor_equation(scale=scale, ridge=0.01, outdir=outdir,
                                   food_df_path=food_df_path, ingnut_df_path=ingnut_df_path,
                                   output_excel_name=output_excel_name, output_matrix_name=output_matrix_name)
    
    # Test 3: Different nutrient combinations
    print("\n" + "="*60)
    print("TEST 3: Different Nutrient Combinations")
    print("="*60)
    
    # Test with only energy and carbohydrates
    energy_carb = ["Energy(kcal)", "Carbohydrate(g)"]
    print(f"\n--- Nutrients: {energy_carb} ---")
    test_yield_factor_equation(nut_cols=energy_carb, scale="none", outdir=outdir,
                              food_df_path=food_df_path, ingnut_df_path=ingnut_df_path,
                              output_excel_name=output_excel_name, output_matrix_name=output_matrix_name)
    
    # Test with only protein and fat
    protein_fat = ["Protein(g)", "Total fat(g)"]
    print(f"\n--- Nutrients: {protein_fat} ---")
    test_yield_factor_equation(nut_cols=protein_fat, scale="none", outdir=outdir,
                              food_df_path=food_df_path, ingnut_df_path=ingnut_df_path,
                              output_excel_name=output_excel_name, output_matrix_name=output_matrix_name)
    
    # Test 4: Different regularization parameters
    print("\n" + "="*60)
    print("TEST 4: Different Regularization Parameters")
    print("="*60)
    
    for ridge in [0.0, 0.01, 0.1, 1.0]:
        print(f"\n--- Ridge: {ridge} ---")
        test_yield_factor_equation(ridge=ridge, scale="none", outdir=outdir,
                                   food_df_path=food_df_path, ingnut_df_path=ingnut_df_path,
                                   output_excel_name=output_excel_name, output_matrix_name=output_matrix_name)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test yield factor prediction with different parameters")
    parser.add_argument("--test-case", default="real", choices=["real", "perfect"], 
                       help="Test case to run (default: real)")
    parser.add_argument("--scale", default="logiqr", choices=["none", "std", "pminmax", "logiqr"], 
                       help="Scaling method (default: logiqr)")
    parser.add_argument("--ridge", type=float, default=0.01, 
                       help="Ridge regularization parameter (default: 0.01)")
    parser.add_argument("--nutrients", default="nut8", 
                       choices=["nut3", "nut4", "nut5", "nut6", "nut8", "energy_carb", "protein_fat"], 
                       help="Nutrient set to use (default: nut8)")
    parser.add_argument("--error", default="l2", choices=["l1", "l2"], 
                       help="Error type: l1 (absolute) or l2 (squared) (default: l2)")
    parser.add_argument("--solver", default="auto", 
                       choices=["auto", "osqp", "ecos", "scs", "clarabel", "piqp"], 
                       help="Optimization solver (default: auto)")
    parser.add_argument("--robust", action="store_true", 
                       help="Use robust Huber loss (only for L2)")
    parser.add_argument("--resolver", default="rule", choices=["rule", "first"], 
                       help="Variant resolution method (default: rule)")
    parser.add_argument("--max-yield-factor", type=float, default=1.0, 
                       help="Maximum allowed yield factor (constraint: 0 < yf <= max_yield_factor) (default: 1.0)")
    parser.add_argument("--method", default="optimization", choices=["optimization", "gpt"], 
                       help="Prediction method: optimization or gpt (default: optimization)")
    parser.add_argument("--api-key", type=str, default=None, 
                       help="OpenAI API key (required for GPT method, or set OPENAI_API_KEY env var)")
    parser.add_argument("--gpt-model", default="gpt-5", choices=["gpt-5", "gpt-4", "gpt-4o"], 
                       help="GPT model to use (default: gpt-5)")
    parser.add_argument("--run-suite", action="store_true", 
                       help="Run full test suite instead of single test")
    parser.add_argument("--outdir", default="./", 
                       help="Output directory for saving results (default: ./)")
    parser.add_argument("--food-df", default="potato_example_foodnut.csv",
                       help="Path to food DataFrame CSV file (default: potato_example_foodnut.csv)")
    parser.add_argument("--ingnut-df", default="potato_example_ingnut.csv",
                       help="Path to ingredient-nutrient DataFrame CSV file (default: potato_example_ingnut.csv)")
    parser.add_argument("--output-excel", default="potato_example_output.xlsx",
                       help="Name of output Excel file (default: potato_example_output.xlsx)")
    parser.add_argument("--output-matrix", default="yield_factors_matrix.npy",
                       help="Name of output numpy matrix file (default: yield_factors_matrix.npy)")
    
    args = parser.parse_args()
    
    if args.run_suite:
        # Run full test suite
        run_test_suite(outdir=args.outdir, food_df_path=args.food_df, ingnut_df_path=args.ingnut_df,
                      output_excel_name=args.output_excel, output_matrix_name=args.output_matrix)
    else:
        # Define nutrient sets
        if args.nutrients == "nut6":
            nut_cols = ["Energy(kcal)", "Carbohydrate(g)", "Protein(g)", "Total fat(g)", "Total sugar(g)", "Sodium(mg)"]
        elif args.nutrients == "nut5":
            nut_cols = C.NUT5  # All nutrients except Total fat
        elif args.nutrients == "nut4":
            nut_cols = C.NUT4  # Energy, Carb, Fat, Protein
        elif args.nutrients == "nut3":
            nut_cols = C.NUT3  # Carb, Fat, Protein
        elif args.nutrients == "energy_carb":
            nut_cols = ["Energy(kcal)", "Carbohydrate(g)"]
        elif args.nutrients == "protein_fat":
            nut_cols = ["Protein(g)", "Total fat(g)"]
        elif args.nutrients == "nut8":
            nut_cols = C.NUT8  # All 8 nutrients
        else:
            nut_cols = C.NUT8  # Default to all NUT8
        
        print(f"Running custom test: test_case={args.test_case}, method={args.method}, "
              f"nutrients={args.nutrients}, max_yield_factor={args.max_yield_factor}")
        if args.method == "optimization":
            print(f"  Optimization params: scale={args.scale}, ridge={args.ridge}, error={args.error}, solver={args.solver}")
        else:
            print(f"  GPT params: model={args.gpt_model}")
        
        test_yield_factor_equation(
            test_case=args.test_case, 
            scale=args.scale, 
            ridge=args.ridge, 
            nut_cols=nut_cols,
            error_type=args.error,
            solver_name=args.solver,
            robust=args.robust,
            resolver=args.resolver,
            max_yield_factor=args.max_yield_factor,
            method=args.method,
            api_key=args.api_key,
            gpt_model=args.gpt_model,
            outdir=args.outdir,
            food_df_path=args.food_df,
            ingnut_df_path=args.ingnut_df,
            output_excel_name=args.output_excel,
            output_matrix_name=args.output_matrix
        )