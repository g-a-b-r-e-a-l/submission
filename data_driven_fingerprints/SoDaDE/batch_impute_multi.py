import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm # For a nice progress bar
from SoDaDE.fingerprint_model.model.decoder import MultiModalRegressionTransformer
from SoDaDE.fingerprint_model.model.config import (VOCAB_SIZE_COLUMNS, MAX_SEQUENCE_LENGTH, 
                       TOKEN_TYPE_VOCAB_SIZE, COLUMN_DICT, TOKEN_TYPE_VOCAB,
                       TRANSFORMER_HIDDEN_DIM, NUM_ATTENTION_HEADS, NUM_TRANSFORMER_LAYERS)
from SoDaDE.fingerprint_model.model.dataset import load_dataset
from SoDaDE.create_plots.collate_finetune import create_fine_collate_fn
from torch.utils.data import DataLoader

class SolventPropertyImputer:
    def __init__(self, model_path, present_values=False, missing_vals=False, vocab_id_lookup=None, device='cpu', 
                 transformer_hidden_dim=None, num_attention_heads=None, num_transformer_layers=None):
        self.model_path = model_path
        self.vocab_id_lookup = vocab_id_lookup
        self.device = device
        self.present_values = present_values
        self.missing_vals = missing_vals
        self.model_config = {
            'chemberta_fp_dim': 384,  # Will be set dynamically
            'column_vocab_size': VOCAB_SIZE_COLUMNS,
            'transformer_hidden_dim': transformer_hidden_dim if transformer_hidden_dim is not None else 128,
            'max_sequence_length': MAX_SEQUENCE_LENGTH,
            'token_type_vocab_size': TOKEN_TYPE_VOCAB_SIZE,
            'num_attention_heads': num_attention_heads if num_attention_heads is not None else 32,
            'num_transformer_layers': num_transformer_layers if num_transformer_layers is not None else 2,
            'dropout_rate': 0.0
        }
        self.collate_fn = create_fine_collate_fn(TOKEN_TYPE_VOCAB)
        
    def load_model(self, model_path):
        """Load model for a specific property if not already loaded"""
        model = MultiModalRegressionTransformer(**self.model_config)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def fill_template_recursively(self, data_path):
        df = pd.read_csv(data_path)
        smiles_column = df['SMILES']
        dataset, _ = load_dataset(data_path, COLUMN_DICT, MAX_SEQUENCE_LENGTH)
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            collate_fn=self.collate_fn, shuffle=False
        )

        filled_dict = {}
        print(self.present_values, 'present vals')
        for idx, batch in enumerate(tqdm(dataloader, desc=f"Imputing values for {data_path}", leave=True)):
            SMILES_fps = batch['SMILES_fps']
            word_tokens_ref = batch['word_tokens_ref']
            values_ref = batch['values_ref']
            token_type_ids = batch['token_type_ids']
            attention_mask = batch['attention_mask']

            # Find all the empty positions in values_ref
            
            if self.missing_vals:
                nan_pos = torch.isnan(values_ref)
                list_positions = nan_pos.tolist()

            else: 
                values_ideal = token_type_ids.clone()
                masked_vals = (token_type_ids == TOKEN_TYPE_VOCAB['MASK_TOKEN'])
                values_ideal[masked_vals] = TOKEN_TYPE_VOCAB['VALUE_TOKEN']
                val_mask = (values_ideal == TOKEN_TYPE_VOCAB['VALUE_TOKEN'])
                list_positions = val_mask.tolist()
            
            list_positions = list_positions[0]

            smiles_id = smiles_column[idx]
            
            # Check if the SMILES ID exists and create a dictionary entry if it doesn't
            # This is moved outside the inner loop to ensure it's run once per batch
            if smiles_id not in filled_dict.keys():
                print(f"Adding new smiles_id {smiles_id}, {idx}")
                filled_dict[smiles_id] = {}

            for index, value_pos in enumerate(list_positions):
                if value_pos: 
                    pos = index
                    token_id = word_tokens_ref[0, pos-1].item()
                    property_label = self.vocab_id_lookup.get(f'{token_id}', None)

                    if property_label is None:
                        print('problem here')
                        continue

                    if property_label not in filled_dict[smiles_id].keys():
                        filled_dict[smiles_id][property_label] = {}
                    
                    if pos not in filled_dict[smiles_id][property_label].keys():
                        filled_dict[smiles_id][property_label][pos] = []

                    model = self.load_model(self.model_path)
                    model.eval()

                    reg_att_mask = ~attention_mask.clone()
                    reg_att_mask[:, 0:pos] = 1 # Set attention mask to focus

                    inference_position = [pos]
    
                    with torch.no_grad():
                        pred = model.generative_inference(
                            TOKEN_TYPE_VOCAB,
                            SMILES_fps,
                            word_tokens_ref,
                            values_ref,
                            token_type_ids,
                            reg_att_mask,
                            inference_position
                        )
                    filled_dict[smiles_id][property_label][pos].append(pred.item())

                    if not self.present_values:
                        values_ref[0, pos] = pred.item() # Update the value in the tensor
                    else: 
                        pass
        print(type(filled_dict), len(filled_dict))

        return filled_dict
        
def tuples_to_lists(obj):
    """
    Recursively convert all tuples in a nested structure (dict, list, etc.)
    into lists so they can be JSON serialised.
    """
    new_dict = {}
    for key, value in obj.items():
        if isinstance(key, tuple):
            new_key = str(key)
            new_dict[new_key] = value
        else:
            new_dict[key] = value    

    return new_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-mp', required=True, 
                        help='Directory containing property models')
    parser.add_argument('--input_file', '-if', required=True,
                        help='Input file with solvent templates')
    parser.add_argument('--output_file_imputed','-of', default='test_predictions.json',
                        help='Output file for imputed results')
    parser.add_argument('--present_values','-pv', action='store_true', 
                        help='Does the data contain values to use as a base?')
    parser.add_argument('--missing_values','-mv', action='store_true', 
                        help='Does the data contain nan values to impute?')
    parser.add_argument('--vocab_id_lookup_file', '-vocab', default='predict_properties/vocab_dict.json',
                        help='File containing vocab ID to property label lookup')
    parser.add_argument('--device', default='cpu',
                        help='Device for model execution (cpu/cuda)')
    parser.add_argument('--hidden_dim', '-hd', type=int, default=64,
                        help='Hidden dimension of the transformer model')
    parser.add_argument('--attention_heads', '-ah', type=int, default=16,
                        help='Number of attention heads in the transformer model')
    parser.add_argument('--transformer_layers', '-tl', type=int, default=5,
                        help='Number of transformer layers in the model')
    args = parser.parse_args()
    
    # Load vocab ID lookup
    with open(args.vocab_id_lookup_file) as f:
        vocab_id_lookup = json.load(f)
    
    # Initialize imputer
    imputer = SolventPropertyImputer(
        model_path=args.model_path,
        present_values=args.present_values,
        missing_vals=args.missing_values,
        vocab_id_lookup=vocab_id_lookup,
        device=args.device,
        transformer_hidden_dim=args.hidden_dim,
        num_attention_heads=args.attention_heads,
        num_transformer_layers=args.transformer_layers
    )
    
    # Fill templates recursively
    filled_data = imputer.fill_template_recursively(args.input_file)
    print(type(filled_data), len(filled_data))
    
    # Convert all tuples to lists for JSON compatibility
    clean_data = tuples_to_lists(filled_data)

    # Save as JSON
    out_name = args.output_file_imputed
    out_path = f"SoDaDE/{out_name}"
    with open(out_path, 'w') as f:
        json.dump(clean_data, f, indent=4)

if __name__ == "__main__":
    main()