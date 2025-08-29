#!/usr/bin/env python3
"""
Data Processing Script for Solvent Property Dataset

This script processes a solvent properties dataset by:
1. Creating train/validation/test splits based on solvent types
2. Transforming the data into a property-value pair format
3. Generating multiple shuffled versions of each solvent's properties
4. Normalizing the data using z-score normalization
5. Saving the processed datasets and normalization statistics

The script is designed to prepare data for machine learning models that need
to handle variable-length sequences of solvent properties.

Usage:
    python data.py

Output Files:
    - train_set.csv: Shuffled training data in property-value pair format
    - val_set.csv: Shuffled validation data in property-value pair format  
    - test_set.csv: Shuffled test data in property-value pair format
    - train_values.csv: Original training data with property columns
    - val_values.csv: Original validation data with property columns
    - test_values.csv: Original test data with property columns
    - normalisation_stats.json: Mean and std deviation for z-score normalization

For more details on the data processing steps, see the melt_shuffle.py module.
"""

import os
import sys
from SoDaDE.fingerprint_model.datasets.melt_shuffle import create_datasets

def main():
    """
    Main function to process the solvent dataset with default parameters.
    
    This function:
    1. Sets up the data processing parameters
    2. Calls the main data processing pipeline
    3. Provides user feedback on the process
    """
    
    print("=" * 60)
    print("SOLVENT PROPERTY DATASET PROCESSING")
    print("=" * 60)
    print()
    
    print("This script will process your solvent properties dataset by:")
    print("• Creating train/validation/test splits based on solvent chemical types")
    print("• Transforming data into property-value pairs for sequence modeling")
    print("• Generating 50 shuffled versions of each solvent's property sequence")
    print("• Applying z-score normalization to training data")
    print("• Saving processed datasets and normalization parameters")
    print()
    
    # Configuration parameters
    config = {
        'file_path': 'SoDaDE/fingerprint_model/datasets/full_extracted_table.csv',
        'rename_solvents': True,
        'num_solvents_per_type': 1,
        'n_shuffles': 50,
        'output_dir': '.',
        'seed': 42
    }
    
    print("Configuration:")
    print(f"• Input file: {config['file_path']}")
    print(f"• Number of shuffles per solvent: {config['n_shuffles']}")
    print(f"• Random seed: {config['seed']}")
    print(f"• Output directory: {config['output_dir']}")
    print()
    
    # Check if input file exists
    if not os.path.exists(config['file_path']):
        print(f"ERROR: Input file '{config['file_path']}' not found!")
        print("Please ensure the dataset file is in the current directory.")
        sys.exit(1)
    
    print("Starting data processing...")
    print("-" * 40)
    
    try:
        # Call the main data processing function
        create_datasets(
            file_path=config['file_path'],
            rename_solvents=config['rename_solvents'],
            num_solvents_per_type=config['num_solvents_per_type'],
            n_shuffles=config['n_shuffles'],
            output_dir=config['output_dir'],
            seed=config['seed']
        )
        
        print("-" * 40)
        print("✅ Data processing completed successfully!")
        print()
        print("Generated files:")
        print("• train_set.csv - Training data with shuffled property sequences")
        print("• val_set.csv - Validation data with shuffled property sequences") 
        print("• test_set.csv - Test data with shuffled property sequences")
        print("• train_values.csv - Original training data with property columns")
        print("• val_values.csv - Original validation data with property columns")
        print("• test_values.csv - Original test data with property columns")
        print("• normalisation_stats.json - Z-score normalization parameters")
        print()
        print("The shuffled datasets contain property-value pairs that can be used")
        print("to train sequence models. The normalization stats are needed to")
        print("properly scale validation/test data and to unnormalize predictions.")
        
    except FileNotFoundError as e:
        print(f"ERROR: Required file not found - {e}")
        print("Please check that all necessary files are present.")
        sys.exit(1)
        
    except Exception as e:
        print(f"ERROR: An error occurred during processing - {e}")
        print("Please check your input data format and try again.")
        sys.exit(1)

def print_data_format_info():
    """
    Print information about the expected data format and processing steps.
    """
    print("\nDATA FORMAT INFORMATION:")
    print("=" * 40)
    print()
    print("Expected Input Format:")
    print("• CSV file with columns: 'solvent', 'SMILES', and property columns")
    print("• Each row represents one solvent with its properties")
    print("• Property values can contain NaN for missing data")
    print()
    print("Data Processing Steps:")
    print("1. Split solvents by chemical type (alkane, ether, ketone, etc.)")
    print("2. Transform wide format (one column per property) to long format")
    print("3. Create property-value pairs: Property_0, Value_0, Property_1, Value_1, etc.")
    print("4. Generate multiple random shuffles of property order for each solvent")
    print("5. Apply z-score normalization: (value - mean) / std_dev")
    print("6. Save processed data and normalization parameters")
    print()
    print("Why Shuffle Properties?")
    print("• Helps models learn that property order shouldn't matter")
    print("• Increases data augmentation for better generalization")
    print("• Useful for sequence-to-sequence or transformer models")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print(__doc__)
        print_data_format_info()
    else:
        main()