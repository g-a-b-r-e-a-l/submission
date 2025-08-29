#!/usr/bin/env python3
"""
Model Evaluation Script for Solvent Property Prediction

This script evaluates the trained SoDaDE model against baseline methods by:
1. Running property imputation on test solvents using shuffled sequences
2. Comparing predictions with Gaussian Process, Random Forest, and averaged property values
3. Supporting two prediction modes:
   - Template: Model uses true property values when building sequences
   - Scratch: Model uses its own predictions when building sequences

The evaluation generates predictions across 50 shuffled property sequences to assess
how prediction performance varies with sequence order.

Usage:
    python eval.py Template    # Use true values as context
    python eval.py Scratch     # Use model predictions as context

Output:
    - test_predictions.json: Detailed prediction results
    - MSE comparison table printed to console
    - Best performing method identified
"""

import sys
import os
from SoDaDE.analyse_results import collate_predictions, calculate_mse_summary

def print_evaluation_info():
    """Print information about the evaluation process and modes."""
    print("EVALUATION MODES:")
    print("=" * 40)
    print()
    print("Template Mode:")
    print("• Model uses true property values as context when predicting")
    print("• Simulates ideal scenario where some properties are known")
    print("• Tests model's ability to fill in missing properties")
    print()
    print("Scratch Mode:")
    print("• Model uses only its own predictions as context")
    print("• More realistic scenario for completely unknown solvents")
    print("• Tests model's ability to predict from minimal information")
    print()
    print("Evaluation Process:")
    print("• Test sequences shuffled 50 times to assess sequence order effects")
    print("• Predictions averaged across all shuffle permutations")
    print("• Results compared against GP, RF, and mean value baselines")
    print("• MSE calculated for each property and overall performance")

def main():
    """Main evaluation function."""
    
    print("=" * 60)
    print("SOLVENT PROPERTY PREDICTION MODEL EVALUATION")
    print("=" * 60)
    print()
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("ERROR: Please specify evaluation mode")
        print("Usage:")
        print("  python eval.py Template    # Use true values as context")
        print("  python eval.py Scratch     # Use model predictions as context")
        print()
        print("For more information, run: python eval.py --help")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode in ['--help', '-h', 'help']:
        print(__doc__)
        print_evaluation_info()
        return
    
    if mode not in ['Template', 'Scratch']:
        print(f"ERROR: Invalid mode '{mode}'")
        print("Valid modes are: 'Template' or 'Scratch'")
        sys.exit(1)
    
    print(f"Running evaluation in {mode} mode...")
    print()
    
    if mode == "Template":
        print("Template Mode: Model will use true property values as context")
        print("• Simulates scenario where some properties are already known")
        print("• Tests model's ability to fill in missing properties accurately")
    else:
        print("Scratch Mode: Model will use its own predictions as context")  
        print("• Simulates realistic scenario for completely unknown solvents")
        print("• Tests model's ability to predict from minimal initial information")
    
    print()
    print("Evaluation will compare SoDaDE against:")
    print("• Gaussian Process (GP) regression baseline")
    print("• Random Forest (RF) regression baseline") 
    print("• Mean property values (AVG) baseline")
    print()
    
    # Check if required files exist
    required_files = [
        'SoDaDE/fingerprint_model/datasets/test_set.csv',
        'SoDaDE/fingerprint_model/datasets/test_values.csv',
        'SoDaDE/fingerprint_model/datasets/normalisation_stats.json',
        'SoDaDE/fingerprint_model/pre-trained_models/val_loss0.1074_DPR_0.1_MP_0.3_DM_64_TL_5_heads_16.pth',
        'SoDaDE/other_property_prediction_methods/data.nosync/predictions_mean.csv',
        'SoDaDE/other_property_prediction_methods/data.nosync/predictions_rf_mean.csv',
        'SoDaDE/create_plots/vocab_dict.json'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("ERROR: Required files not found:")
        for file in missing_files:
            print(f"  • {file}")
        print()
        print("Please ensure all required files are present before running evaluation.")
        sys.exit(1)
    
    try:
        print("Step 1: Loading baseline predictions and test data...")
        print("-" * 50)
        
        # Note: The mode parameter would be used to modify the imputation behavior
        # For now, the collate_predictions function runs with default settings
        # You would need to modify analyse_results.py to accept and use the mode parameter
        
        if mode == "Scratch":
            AVG_df, RF_df, GP_df, SoDaDE_df, test_df, norm_dict = collate_predictions(mode)
            

        elif mode == "Template":
            AVG_df, RF_df, GP_df, SoDaDE_df, test_df, norm_dict = collate_predictions(mode)
        
        
        print("Step 2: Processing predictions and calculating MSE metrics...")
        print("-" * 50)
        
        mse_summary_df = calculate_mse_summary(
            AVG_df, RF_df, GP_df, SoDaDE_df, test_df, norm_dict
        )
        
        # Add best model identification
        mse_summary_df['Best_Method'] = mse_summary_df.drop('Best_Method', axis=1, errors='ignore').idxmin(axis=1)
        
        overall_best = mse_summary_df.loc['Average MSE', 'Best_Method']
        
        print("Step 3: Evaluation Results")
        print("-" * 50)
        print()
        print("Mean Squared Error (MSE) Comparison:")
        print("=" * 60)
        
        # Format the results table nicely
        display_df = mse_summary_df.drop('Best_Method', axis=1, errors='ignore')
        
        # Round values for better display
        display_df = display_df.round(6)
        
        print(display_df.to_string())
        print()
        
        print("Best Method by Property:")
        print("-" * 30)
        best_methods = mse_summary_df['Best_Method']
        for prop, method in best_methods.items():
            print(f"{prop:<25} {method}")
        
        print()
        print(f"Overall Best Method: {overall_best}")
        print(f"(Based on Average MSE across all properties)")
        
        # Save detailed results
        output_file = f"evaluation_results_{mode.lower()}.csv"
        mse_summary_df.to_csv(output_file)
        print(f"\nDetailed results saved to: {output_file}")
        print(f"Prediction details saved to: predict_properties/test_predictions.json")
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"ERROR: Required file not found - {e}")
        print("Please check that all necessary files are present.")
        sys.exit(1)
        
    except Exception as e:
        print(f"ERROR: Evaluation failed - {e}")
        print("Please check your data files and model configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()