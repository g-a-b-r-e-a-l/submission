import json
import os
import pandas as pd
import numpy as np
from rdkit import Chem

# Import the SolventPropertyImputer class from batch_impute_multi
from SoDaDE.batch_impute_multi import SolventPropertyImputer, tuples_to_lists

def run_imputation_and_get_predictions(model_path, test_file, mode, output_json="test_predictions.json",
                                     vocab_file="predict_properties/vocab_dict.json",
                                     hidden_dim=64, attention_heads=16, transformer_layers=5,
                                     device="cpu"):
    """
    Run imputation on test dataset and return averaged predictions as a DataFrame.
    
    Args:
        model_path: Path to the trained model file
        test_file: Path to the test dataset CSV file
        output_json: Name of the output JSON file
        vocab_file: Path to the vocabulary lookup file
        hidden_dim: Hidden dimension of the model
        attention_heads: Number of attention heads
        transformer_layers: Number of transformer layers
        device: Device to run on (cpu/cuda)
    
    Returns:
        pd.DataFrame: DataFrame with SMILES as first column and averaged predictions for each property
    """
    
    # Load vocab ID lookup
    with open(vocab_file) as f:
        vocab_id_lookup = json.load(f)
    
    # Initialize imputer based on mode
    if mode == "Template":
        present_values = True
        missing_vals = False
    elif mode == 'Scratch':
        present_values = False
        missing_vals = False
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'Template' or 'Scratch'")
    
    # Initialize imputer
    imputer = SolventPropertyImputer(
        model_path=model_path,
        present_values=present_values,
        missing_vals=missing_vals,
        vocab_id_lookup=vocab_id_lookup,
        device=device,
        transformer_hidden_dim=hidden_dim,
        num_attention_heads=attention_heads,
        num_transformer_layers=transformer_layers
    )
    
    # Fill templates recursively
    filled_data = imputer.fill_template_recursively(test_file)
    
    # Convert all tuples to lists for JSON compatibility
    clean_data = tuples_to_lists(filled_data)
    
    # Save as JSON
    if mode == "Template":
        out_path = f"SoDaDE/{output_json}"
    else:  # Scratch mode
        out_path = f"predict_properties/{output_json}"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(clean_data, f, indent=4)
    
    # Process results into DataFrame (same logic as before)
    data = []
    
    for smiles, properties_dict in clean_data.items():
        row = {'SMILES': smiles}
        
        for property_name, positions_dict in properties_dict.items():
            # Collect all predictions for this property
            all_predictions = []
            for position, predictions_list in positions_dict.items():
                all_predictions.extend(predictions_list)
            
            if all_predictions:
                # Calculate average prediction
                row[property_name] = np.mean(all_predictions)
        
        data.append(row)
    
    return pd.DataFrame(data)

def collate_predictions(mode):

    #load GP results
    GP_results_file = 'SoDaDE/other_property_prediction_methods/data.nosync/predictions_mean.csv'
    GP_df = pd.read_csv(GP_results_file)
    GP_df.columns = [col.split(" ")[-1] for col in GP_df.columns]

    #load RF results
    RF_results_file = 'SoDaDE/other_property_prediction_methods/data.nosync/predictions_rf_mean.csv'
    RF_df = pd.read_csv(RF_results_file)
    RF_df.columns = [col.split(" ")[-1] for col in RF_df.columns]

    #load Test values
    test_file = 'SoDaDE/fingerprint_model/datasets/test_values.csv'
    test_df = pd.read_csv(test_file)
    test_df.drop(columns='solvent', inplace=True)

    #load normalisation parameters
    norm_path = 'SoDaDE/fingerprint_model/datasets/normalisation_stats.json'
    with open(norm_path, 'r') as f:
        norm_dict = json.load(f)

    #run SoDaDE model predictions
    model_path = 'SoDaDE/fingerprint_model/pre-trained_models/val_loss0.1074_DPR_0.1_MP_0.3_DM_64_TL_5_heads_16.pth'
    test_file = 'SoDaDE/fingerprint_model/datasets/test_set.csv'

    SoDaDE_df = run_imputation_and_get_predictions(model_path, test_file, mode, output_json="test_predictions.json",
                                     vocab_file="SoDaDE/create_plots/vocab_dict.json",
                                     hidden_dim=64, attention_heads=16, transformer_layers=5,
                                     device="cpu")

    AVG_df = test_df.copy()
    for property, stats in norm_dict.items():
        AVG_df[property] = stats['mean']

    
    
    return AVG_df, RF_df, GP_df, SoDaDE_df, test_df, norm_dict

# RDKit is required for canonicalising SMILES. Install with: pip install rdkit

def calculate_mse_summary(AVG_df, RF_df, GP_df, SoDaDE_df, test_df, norm_dict):
    """
    Canonicalises SMILES, unnormalises predictions and test data, 
    and calculates MSE for each method.

    Args:
        GP_df (pd.DataFrame): Gaussian Process model predictions.
        RF_df (pd.DataFrame): Random Forest model predictions.
        AVG_df (pd.DataFrame): Mean value predictions (already unnormalised).
        SoDaDE_df (pd.DataFrame): SoDaDE model predictions.
        test_df (pd.DataFrame): Ground truth values (normalised).
        norm_dict (dict): Dictionary with normalisation statistics ('mean', 'std').

    Returns:
        pd.DataFrame: A DataFrame with methods as columns, properties as rows,
                      and a final row for the average MSE per method.
    """
    
    # --- 1. Helper function for canonicalisation ---
    def canonicalize_smiles(smiles):
        """Generates canonical SMILES, returns original if invalid."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        return smiles

    # --- 2. Define DataFrame groups ---
    # Per your request, 'test_df' is now included in the unnormalisation process.
    dfs_to_unnormalize = {'GP': GP_df, 'RF': RF_df, 'SoDaDE': SoDaDE_df, 'Test': test_df}
    
    # All DataFrames for general processing (alignment, etc.)
    all_dfs = {**dfs_to_unnormalize, 'AVG': AVG_df}
    
    # Prediction DataFrames for the final MSE calculation.
    prediction_dfs = {'AVG': AVG_df, 'RF': RF_df, 'GP': GP_df, 'SoDaDE': SoDaDE_df}

    # --- 3. Canonicalise SMILES across all DataFrames ---
    print("Canonicalising SMILES...")
    for name, df in all_dfs.items():
        if 'SMILES' not in df.columns:
            df.reset_index(inplace=True)
        df['SMILES'] = df['SMILES'].apply(canonicalize_smiles)

    # --- 4. Unnormalise prediction AND test data ---
    print("Unnormalising dataframes...")
    # Get property columns from the original test_df structure
    properties = [col for col in test_df.columns if col not in ['SMILES']]
    
    for name, df in dfs_to_unnormalize.items():
        for prop in properties:
            if prop in norm_dict and prop in df.columns:
                std = norm_dict[prop]['std']
                mean = norm_dict[prop]['mean']
                # Reverse z-score normalisation: x = (z * std) + mean
                df[prop] = (df[prop] * std) + mean
    
    # --- 5. Align DataFrames on common SMILES ---
    for name, df in all_dfs.items():
        df.drop_duplicates(subset='SMILES', inplace=True)
        df.set_index('SMILES', inplace=True)
        
    common_smiles = all_dfs['Test'].index
    for name, df in prediction_dfs.items():
        common_smiles = common_smiles.intersection(df.index)

    true_values = all_dfs['Test'].loc[common_smiles].sort_index()
    for name, pred_df in prediction_dfs.items():
        prediction_dfs[name] = pred_df.loc[common_smiles].sort_index()

    # --- 6. Calculate MSE ---
    print("Calculating Mean Squared Error...")
    mse_results = {}
    
    for method, pred_df in prediction_dfs.items():
        squared_errors = (pred_df[properties] - true_values[properties])**2
        mse_results[method] = squared_errors.mean()

    # --- 7. Format the final DataFrame ---
    results_df = pd.DataFrame(mse_results)
    results_df.loc['Average MSE'] = results_df.mean()
    
    return results_df

if __name__ == "__main__":
    # 1. Collate all the data and predictions
    mode = "Template"
    AVG_df, RF_df, GP_df, SoDaDE_df, test_df, norm_dict = collate_predictions(mode)
    
    # 2. Process the data and calculate the MSE summary
    mse_summary_df = calculate_mse_summary(
        AVG_df, RF_df, GP_df, SoDaDE_df, test_df, norm_dict
    )
    
    # 3. Find the best model for each row and the overall best model
    # Get the column name with the minimum value for each row (property)
    mse_summary_df['Lowest MSE Model'] = mse_summary_df.idxmin(axis=1)
    
    # Determine the overall best model based on the 'Average MSE' row
    overall_best_model = mse_summary_df.loc['Average MSE', 'Lowest MSE Model']
    
    # 4. Print the final results
    print("\n--- MSE Summary ---")
    print(mse_summary_df)
    print(f"\nOverall Best Model (based on Average MSE): {overall_best_model}")