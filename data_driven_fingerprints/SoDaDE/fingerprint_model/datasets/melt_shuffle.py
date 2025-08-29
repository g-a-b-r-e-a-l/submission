import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import math
import argparse
from SoDaDE.fingerprint_model.datasets.prepare_data import create_train_val_test_split


def transform_dataframe(df):
    """
    Transforms the input DataFrame by unpivoting property columns and their values
    into a key-value pair format in each row.

    Args:
        df (pd.DataFrame): The input DataFrame with 'solvent', 'solvent smiles',
                           and various property columns.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    # Identify the 'id_vars' (columns to keep as identifiers)
    id_vars = ['solvent', 'SMILES']
    print(df.columns)

    # Melt the DataFrame to unpivot the property columns
    df_melted = df.melt(id_vars=id_vars, var_name='Property', value_name='Value')

    # Debug: Check if any properties have all NaN values
    value_counts = df_melted.groupby('Property')['Value'].apply(lambda x: x.notna().sum())
    print("Non-NaN value counts per property:")
    print(value_counts)

    # Create a new column 'col_idx' to interleave Property and Value
    df_melted['col_idx'] = df_melted.groupby(id_vars).cumcount()

    # Pivot the table to get 'Property' and 'Value' side-by-side
    df_transformed = df_melted.pivot_table(
        index=id_vars,
        columns='col_idx',
        values=['Property', 'Value'],
        aggfunc='first'
    )

    # Flatten the MultiIndex columns
    df_transformed.columns = [f'{col[0]}_{col[1]}' for col in df_transformed.columns]

    # Determine the number of property columns from the original data
    num_properties = len([col for col in df.columns if col not in id_vars])

    # Create the expected column names
    expected_columns = []
    for i in range(num_properties):
        expected_columns.append(f'Property_{i}')
        expected_columns.append(f'Value_{i}')

    # Add missing columns with NaN values
    for col in expected_columns:
        if col not in df_transformed.columns:
            df_transformed[col] = np.nan
            print(f"Added missing column: {col}")

    print("Available columns:", df_transformed.columns.tolist())
    print("Expected columns:", expected_columns)

    # Reorder columns to have Property and Value pairs
    df_transformed = df_transformed[expected_columns]

    # Reset index to bring 'solvent' and 'SMILES' back as columns
    df_transformed = df_transformed.reset_index()

    return df_transformed

def shuffle_column_pairs(df, n_shuffles):
    """
    For each row, generates a specified number of unique random shuffles
    of its property-value pairs. This helps ensure better distribution
    compared to generating permutations.

    Args:
        df (pd.DataFrame): The input DataFrame with 'solvent', 'SMILES',
                           and interleaved 'Property_i' and 'Value_i' columns.
        n_shuffles (int): The number of unique random shuffles to generate
                          for each original row.

    Returns:
        pd.DataFrame: A new DataFrame containing the generated shuffled rows.
                      The original rows are not included.
    """
    if n_shuffles <= 0:
        print("Warning: Number of shuffles must be positive. Returning an empty DataFrame.")
        return pd.DataFrame(columns=df.columns)

    generated_rows = []

    # Define the fixed identifier columns
    id_cols = ['solvent', 'SMILES']

    # Dynamically find all property-value column indices
    pair_indices = sorted([int(col.split('_')[1]) for col in df.columns if col.startswith('Property_')])

    if not pair_indices:
        print("Error: No 'Property_i' columns found. Returning an empty DataFrame.")
        return pd.DataFrame(columns=df.columns)

    num_pairs = len(pair_indices)

    # Iterate over each original row in the DataFrame
    for _, original_row in df.iterrows():
        # Collect all property-value pairs for the current row
        property_value_pairs = []
        for i in pair_indices:
            prop_col = f'Property_{i}'
            val_col = f'Value_{i}'
            prop = original_row.get(prop_col, np.nan)
            val = original_row.get(val_col, np.nan)
            property_value_pairs.append((prop, val))

        # Use a set to keep track of already generated shuffled sequences
        # to ensure uniqueness for the current row's shuffles
        unique_shuffles_for_row = set()
        shuffles_count = 0

        # Continue generating shuffles until we reach n_shuffles or
        # exhaust all possible unique shuffles (if num_pairs is small)
        max_possible_shuffles = math.factorial(num_pairs)

        while shuffles_count < n_shuffles and len(unique_shuffles_for_row) < max_possible_shuffles:
            temp_pairs = list(property_value_pairs) # Create a mutable copy
            random.shuffle(temp_pairs)

            # Convert to tuple for hashing in the set
            shuffled_tuple = tuple(temp_pairs)

            if shuffled_tuple not in unique_shuffles_for_row:
                unique_shuffles_for_row.add(shuffled_tuple)
                shuffles_count += 1

                new_row_data = {col: original_row[col] for col in id_cols}

                # Populate the new row with the shuffled property-value pairs
                for i, (prop, val) in enumerate(temp_pairs):
                    new_row_data[f'Property_{i}'] = prop
                    new_row_data[f'Value_{i}'] = val

                generated_rows.append(new_row_data)

        if shuffles_count < n_shuffles:
             print(f"Warning: Could only generate {shuffles_count} unique shuffles for a row with {num_pairs} pairs (original row index: {_}). "
                   f"Requested {n_shuffles}.")

    # Shuffle the entire list of generated rows before creating the DataFrame (optional, but good for overall randomness)
    random.shuffle(generated_rows)

    # Create the final DataFrame from the list of generated row dictionaries
    if not generated_rows:
        return pd.DataFrame(columns=df.columns)

    # Ensure column order is maintained based on the original DataFrame structure
    # This assumes that the original DataFrame has columns like Property_0, Value_0, Property_1, Value_1 etc.
    column_order = id_cols
    for i in pair_indices:
        column_order.append(f'Property_{i}')
        column_order.append(f'Value_{i}')

    final_df = pd.DataFrame(generated_rows)
    # Reindex to ensure consistent column order with original df
    final_df = final_df.reindex(columns=column_order)


    return final_df

def create_datasets(file_path, rename_solvents, 
                    num_solvents_per_type, n_shuffles, output_dir, seed=42):
    
    val_solvent_types = ['alkane', 'ether', 'ketone', 'ester',
        'nitrile',  'amide', 'carboxylic_acid',
        'monohydric_alcohol', 'polyhydric_alcohol']
    test_solvent_types = ['alkane', 'ether', 'ketone', 'ester','monohydric_alcohol']
    train_df, val_df, test_df, stats, test_solvents_info = create_train_val_test_split(
        file_path,
        rename_solvents=rename_solvents,
        method='z_score',
        val_solvent_types=val_solvent_types,
        test_solvent_types=test_solvent_types,
        num_solvents_per_type=num_solvents_per_type,
        seed=seed
    )

    train_melt = transform_dataframe(train_df)
    val_melt = transform_dataframe(val_df)
    test_melt = transform_dataframe(test_df)
    
    train_shuffle = shuffle_column_pairs(train_melt, n_shuffles)
    val_shuffle = shuffle_column_pairs(val_melt, n_shuffles)
    test_shuffle = shuffle_column_pairs(test_melt, n_shuffles)

    print(f"Training set shape: {train_shuffle.shape}, Val shape: {val_shuffle.shape}, Test set shape: {test_shuffle.shape}")

    # Save CSVs
    train_csv_path = f"SoDaDE/fingerprint_model/datasets/train_set.csv"
    val_csv_path = f"SoDaDE/fingerprint_model/datasets/val_set.csv"
    test_csv_path = f"SoDaDE/fingerprint_model/datasets/test_set.csv"

    train_values_path = f"SoDaDE/fingerprint_model/datasets/train_values.csv"
    val_values_path = f"SoDaDE/fingerprint_model/datasets/val_values.csv"
    test_values_path = f"SoDaDE/fingerprint_model/datasets/test_values.csv"

    train_shuffle.to_csv(train_csv_path, index=False)
    val_shuffle.to_csv(val_csv_path, index=False)
    test_shuffle.to_csv(test_csv_path, index=False)

    train_df.to_csv(train_values_path, index=False)
    val_df.to_csv(val_values_path, index=False)
    test_df.to_csv(test_values_path, index=False)

    # Save normalisation stats to JSON
    norm_json_path = f"SoDaDE/fingerprint_model/datasets/normalisation_stats.json"
    with open(norm_json_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Train set has {train_df.shape[0]} solvents, Val set has {val_df.shape[0]} solvents and test has {test_df.shape[0]} solvents")
    print(f"Saved:\n - {train_csv_path}\n - {val_csv_path}\n- {test_csv_path}\n - {norm_json_path}")
    print(test_solvents_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create shuffled datasets and save normalisation stats.")
    parser.add_argument("--file_path", type=str, help="Path to the input dataset file", default='full_extracted_table.csv')
    parser.add_argument("--rename_solvents", action="store_true", help="Whether to rename solvents")

    parser.add_argument("--num_solvents_per_type", type=int, default=1, help="Number of solvents to select from each type")
    parser.add_argument("--n_shuffles", type=int, default=5, help="Number of shuffles per row")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    create_datasets(
        file_path=args.file_path,
        rename_solvents=args.rename_solvents,
        num_solvents_per_type=args.num_solvents_per_type,
        n_shuffles=args.n_shuffles,
        output_dir=args.output_dir,
        seed=args.seed
    )


