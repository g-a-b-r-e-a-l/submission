#!/usr/bin/env python3
"""
Model Training Script for Solvent Property Prediction

This script trains a MultiModal Regression Transformer model for predicting
solvent properties from shuffled property-value sequences. The model uses:
- ChemBERTa embeddings for molecular fingerprints
- Transformer architecture for sequence modeling
- Masked language modeling during training
- Early stopping based on validation loss

The script is configured with optimal hyperparameters determined through
hyperparameter tuning experiments.

Usage:
    python train.py

Model Architecture:
    - Multi-modal transformer with attention mechanism
    - Combines molecular embeddings with property sequences
    - Dropout and masking for regularization
    - AdamW optimizer with learning rate scheduling

Output Files:
    - Best model saved in model/saved_models/ directory
    - Training/validation loss curves in model/Loss_over_time.csv

For more details on the model architecture, see the model/ directory modules.
"""

import os
import sys
from SoDaDE.train_model import train

def print_training_info():
    """
    Print information about the training process and model architecture.
    """
    print("MODEL TRAINING INFORMATION:")
    print("=" * 50)
    print()
    print("Model Architecture:")
    print("• MultiModal Regression Transformer")
    print("• ChemBERTa molecular fingerprint embeddings") 
    print("• Multi-head self-attention layers")
    print("• Masked language modeling for property sequences")
    print("• Regression head for property value prediction")
    print()
    print("Training Process:")
    print("• AdamW optimizer with learning rate scheduling")
    print("• Early stopping based on validation loss")
    print("• Dropout and token masking for regularization")
    print("• Best model automatically saved during training")
    print()
    print("Expected Input Data:")
    print("• Training data: Shuffled property-value sequences")
    print("• Validation data: Shuffled property-value sequences")
    print("• Data should be normalized (z-score) before training")
    print("• SMILES strings for molecular fingerprint generation")

def main_training():
    """
    Main function to train the solvent property prediction model.
    
    This function calls the training pipeline with optimal hyperparameters
    determined through experimentation and hyperparameter tuning.
    """
    
    print("=" * 60)
    print("SOLVENT PROPERTY PREDICTION MODEL TRAINING")
    print("=" * 60)
    print()
    
    print("This script will train a transformer model for solvent property prediction using:")
    print("• MultiModal Regression Transformer architecture")
    print("• ChemBERTa molecular embeddings")
    print("• Masked language modeling on property sequences")
    print("• Optimal hyperparameters from tuning experiments")
    print()
    
    # Optimal hyperparameters (you can modify these as needed)
    optimal_params = {
        'number_of_epochs': 100,
        'learning_rate': 0.001,
        'batch_size': 32,
        'shuffle': True,
        'data_path': "SoDaDE/fingerprint_model/datasets/train_set.csv",           # Training data from data.py
        'val_path': "SoDaDE/fingerprint_model/datasets/val_set.csv",              # Validation data from data.py
        'masking_probability': 0.3,            # Optimal masking probability
        'dropout_rate': 0.1,                    # Optimal dropout rate
        'transformer_layers': 5,                # Optimal number of layers
        'model_dimension': 64,                 # Optimal model dimension
        'attention_heads': 16                   # Optimal attention heads
    }
    
    print("Optimal Training Configuration:")
    print(f"• Number of epochs: {optimal_params['number_of_epochs']}")
    print(f"• Learning rate: {optimal_params['learning_rate']}")
    print(f"• Batch size: {optimal_params['batch_size']}")
    print(f"• Masking probability: {optimal_params['masking_probability']}")
    print(f"• Dropout rate: {optimal_params['dropout_rate']}")
    print(f"• Transformer layers: {optimal_params['transformer_layers']}")
    print(f"• Model dimension: {optimal_params['model_dimension']}")
    print(f"• Attention heads: {optimal_params['attention_heads']}")
    print()
    
    # Check if required data files exist
    required_files = [optimal_params['data_path'], optimal_params['val_path']]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("ERROR: Required data files not found:")
        for file in missing_files:
            print(f"  • {file}")
        print()
        print("Please run 'python data.py' first to generate the required datasets.")
        sys.exit(1)
    
    # Check if model directory exists
    model_dir = "SoDaDE/fingerprint_model/model"
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory '{model_dir}' not found!")
        print("Please ensure the model directory with all required modules is present.")
        sys.exit(1)
    
    required_model_files = [
        "SoDaDE/fingerprint_model/model/dataset.py",
        "SoDaDE/fingerprint_model/model/collate.py", 
        "SoDaDE/fingerprint_model/model/predict_values.py",
        "SoDaDE/fingerprint_model/model/decoder.py",
        "SoDaDE/fingerprint_model/model/config.py"
    ]
    
    missing_model_files = [f for f in required_model_files if not os.path.exists(f)]
    if missing_model_files:
        print("WARNING: Some model files may be missing:")
        for file in missing_model_files:
            print(f"  • {file}")
        print("Training may fail if these files are required.")
        print()
    
    print("Starting model training. Best performing models on the validation dataset are " \
    "saved in the folder 'saved_models' within the 'model' directory")
    print("-" * 50)
    print()
    
    try:
        # Call the main training function with optimal parameters
        result = train(
            number_of_epochs=optimal_params['number_of_epochs'],
            learning_rate=optimal_params['learning_rate'],
            batch_size=optimal_params['batch_size'],
            shuffle=optimal_params['shuffle'],
            data_path=optimal_params['data_path'],
            val_path=optimal_params['val_path'],
            masking_probability=optimal_params['masking_probability'],
            dropout_rate=optimal_params['dropout_rate'],
            transformer_layers=optimal_params['transformer_layers'],
            model_dimension=optimal_params['model_dimension'],
            attention_heads=optimal_params['attention_heads']
        )
        
        print()
        print("-" * 50)
        print("✅ Model training completed successfully!")
        print()
        print("Generated files:")
        print("• Best model saved in: SoDaDE/fingerprint_model/saved_models_from_training")
        print("• Training curves saved in: SoDaDE/fingerprint_model/Loss_over_time.csv")
        print()
        print("The best model (lowest validation loss) has been automatically saved")
        print("with a filename indicating its performance and hyperparameters.")
        print("You can use this model for inference on new solvent data.")
        
    except FileNotFoundError as e:
        print(f"ERROR: Required file not found - {e}")
        print("Please check that all necessary files and modules are present.")
        sys.exit(1)
        
    except ImportError as e:
        print(f"ERROR: Failed to import required module - {e}")
        print("Please ensure all model modules are present in the model/ directory.")
        sys.exit(1)
        
    except Exception as e:
        print(f"ERROR: Training failed - {e}")
        print("Please check your data format and model configuration.")
        sys.exit(1)

def print_custom_params_info():
    """
    Print information about how to modify hyperparameters.
    """
    print("\nCUSTOM HYPERPARAMETER INFORMATION:")
    print("=" * 50)
    print()
    print("To use different hyperparameters, modify the 'optimal_params' dictionary")
    print("in the main_training() function of this script.")
    print()
    print("Key hyperparameters to consider:")
    print("• number_of_epochs: More epochs = longer training, risk of overfitting")
    print("• learning_rate: Higher = faster learning but less stable")
    print("• batch_size: Larger = more stable gradients, needs more memory")
    print("• model_dimension: Larger = more capacity, needs more memory")
    print("• transformer_layers: More layers = more capacity, slower training")
    print("• masking_probability: Higher = more regularization")
    print("• dropout_rate: Higher = more regularization, may hurt performance")
    print()
    print("The current values are optimized based on hyperparameter tuning.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print(__doc__)
        print_training_info()
        print_custom_params_info()
    elif len(sys.argv) > 1 and sys.argv[1] in ['--info', '-i', 'info']:
        print_training_info()
    else:
        main_training()