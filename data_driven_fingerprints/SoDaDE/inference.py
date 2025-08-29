import argparse
import textwrap
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from SoDaDE.fingerprint_model.model.dataset import load_dataset
from SoDaDE.fingerprint_model.model.collate import create_collate_fn
from SoDaDE.fingerprint_model.model.decoder import MultiModalRegressionTransformer

from SoDaDE.fingerprint_model.model.config import (VOCAB_SIZE_COLUMNS, TRANSFORMER_HIDDEN_DIM, 
                    MAX_SEQUENCE_LENGTH, TOKEN_TYPE_VOCAB_SIZE, 
                    NUM_ATTENTION_HEADS, NUM_TRANSFORMER_LAYERS,  
                    COLUMN_DICT, TOKEN_TYPE_VOCAB)


def run_generative_inference(model, dataloader, positions_to_predict, num_samples=None):
    """
    Run generative inference on a dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader with data for inference
        positions_to_predict: List of sequence positions to predict
        num_samples: Number of samples to process (None for all)
    
    Returns:
        List of predicted values for each sample
    """
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(dataloader, desc="Running generative inference"):
            # Extract batch data (adjust keys based on your collate function output)
            token_type_vocab = TOKEN_TYPE_VOCAB
            SMILES_fps = batch['SMILES_fps']
            word_tokens_ref = batch['word_tokens_ref']
            values_ref = batch['values_ref']
            token_type_ids = batch['token_type_ids']
            attention_mask = batch['attention_mask']

            
            # Run generative inference
            predictions = model.generative_inference(
                token_type_vocab=token_type_vocab,
                SMILES_fps=SMILES_fps,
                word_tokens_ref=word_tokens_ref,
                values_ref=values_ref,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                positions_to_predict=positions_to_predict
            )
            
            all_predictions.append(predictions)
            sample_count += predictions.shape[0]
            
            # Stop if we've processed enough samples
            if num_samples is not None and sample_count >= num_samples:
                break
    
    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    
    if num_samples is not None:
        all_predictions = all_predictions[:num_samples]
    
    return all_predictions


def load_model(model_path, chemberta_dimension, dropout_rate):
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the saved model
        chemberta_dimension: Dimension of ChemBERTa features
        dropout_rate: Dropout rate used during training
    
    Returns:
        Loaded model
    """
    model = MultiModalRegressionTransformer(
        chemberta_fp_dim=chemberta_dimension,
        column_vocab_size=VOCAB_SIZE_COLUMNS,
        transformer_hidden_dim=64,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        token_type_vocab_size=TOKEN_TYPE_VOCAB_SIZE,
        num_attention_heads=16,
        num_transformer_layers=4,
        dropout_rate=0
    )
    
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    return model


def main(
    model_path: str,
    data_path: str,
    inference_positions: list,
    batch_size: int = 16,
    num_samples: int = None,
    masking_probability: float = 0.0,  # Usually no masking during inference
    dropout_rate: float = 0.3,
    output_path: str = None
):
    """
    Main inference function.
    
    Args:
        model_path: Path to trained model
        data_path: Path to data file for inference
        inference_positions: List of positions to predict
        batch_size: Batch size for inference
        num_samples: Number of samples to process (None for all)
        masking_probability: Masking probability (usually 0 for inference)
        dropout_rate: Dropout rate used during training
        output_path: Custom output path for predictions
    """
    
    # Load dataset
    print(f"Loading dataset from {data_path}")
    dataset, chemberta_dimension = load_dataset(data_path, COLUMN_DICT, MAX_SEQUENCE_LENGTH)
    
    # Create collate function
    configured_collate_fn = create_collate_fn(TOKEN_TYPE_VOCAB, 0)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Usually no shuffling for inference
        collate_fn=configured_collate_fn
    )
    
    # Load model
    model = load_model(model_path, chemberta_dimension, dropout_rate)
    
    print(f"Running generative inference on positions: {inference_positions}")
    print(f"Processing {len(dataset)} samples" + (f" (limited to {num_samples})" if num_samples else ""))
    
    # Run inference
    predictions = run_generative_inference(
        model=model,
        dataloader=dataloader,
        positions_to_predict=inference_positions,
        num_samples=num_samples
    )
    
    print(f"Generated predictions for {predictions.shape[0]} samples")
    print(f"Predictions shape: {predictions.shape}")
    
    # Save predictions
    if output_path is None:
        output_path = f"inference_pred_pos{inference_positions}_{data_path}.pt"
    
    torch.save(predictions, output_path)
    print(f"Predictions saved to: {output_path}")
    
    # Optionally save as text file for easy inspection
    text_output_path = output_path.replace('.pt', '.txt')
    with open(text_output_path, 'w') as f:
        f.write(f"Predictions for positions {inference_positions}\n")
        f.write(f"Shape: {predictions.shape}\n\n")
        for i, pred in enumerate(predictions):
            f.write(f"Sample {i}: {pred.tolist()}\n")
    
    print(f"Human-readable predictions saved to: {text_output_path}")
    
    return predictions


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run generative inference with trained transformer model.",
        epilog=textwrap.dedent(
            """
            Example usage:
                # Basic inference
                python inference.py -mp "model.pth" -dp "test_data.csv" -ip "[2,5,8]"
                
                # Inference with custom settings
                python inference.py -mp "model.pth" -dp "test_data.csv" -ip "[0,1,3,7]" -bs 32 -ns 500 -op "my_predictions.pt"
                
            Positions format:
                -ip "[2,5,8]" means predict values at sequence positions 2, 5, and 8
                The model will predict these sequentially in sorted order
            """
        ),
    )
    
    # Required arguments
    argparser.add_argument("-mp", "--model_path", type=str, required=True, 
                          help="Path to saved model file (.pth)")
    argparser.add_argument("-dp", "--data_path", type=str, required=True,
                          help="Path to data file for inference (.csv)")
    argparser.add_argument("-ip", "--inference_positions", type=str, required=True,
                          help="List of positions to predict (as string, e.g., '[2,5,8]')")
    
    # Optional arguments
    argparser.add_argument("-bs", "--batch_size", type=int, default=16,
                          help="Batch size for inference (default: 16)")
    argparser.add_argument("-ns", "--num_samples", type=int, default=None,
                          help="Number of samples to process (default: all)")
    argparser.add_argument("-mp_prob", "--masking_probability", type=float, default=0.0,
                          help="Masking probability (default: 0.0 for inference)")
    argparser.add_argument("-dr", "--dropout_rate", type=float, default=0.3,
                          help="Dropout rate used during training (default: 0.3)")
    argparser.add_argument("-op", "--output_path", type=str, default=None,
                          help="Custom output path for predictions (default: auto-generated)")
    
    args = argparser.parse_args()
    
    # Parse inference positions
    try:
        inference_positions = eval(args.inference_positions)
        if not isinstance(inference_positions, list):
            raise ValueError("Inference positions must be a list")
        if not all(isinstance(pos, int) for pos in inference_positions):
            raise ValueError("All positions must be integers")
    except Exception as e:
        raise ValueError(f"Invalid format for inference positions: {e}")
    
    # Run inference
    predictions = main(
        model_path=args.model_path,
        data_path=args.data_path,
        inference_positions=inference_positions,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        masking_probability=args.masking_probability,
        dropout_rate=args.dropout_rate,
        output_path=args.output_path
    )