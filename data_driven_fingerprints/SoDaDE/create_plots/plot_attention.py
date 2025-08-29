import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from typing import List, Dict, Optional, Union

# --- Modified Masked Multi-Head Self-Attention Block ---
class MaskedMultiHeadSelfAttentionBlockWithWeights(nn.Module):
    """
    Modified version that can return attention weights
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ffn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, tgt_mask: torch.Tensor = None, 
                tgt_key_padding_mask: torch.Tensor = None, return_attention: bool = False):
        if return_attention:
            attn_output, attn_weights = self.self_attn(
                query=tgt,
                key=tgt,
                value=tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=False,
                average_attn_weights=False  # Keep all heads separate
            )
        else:
            attn_output, _ = self.self_attn(
                query=tgt,
                key=tgt,
                value=tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=False
            )
            attn_weights = None
            
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        ff_output = self.linear2(self.dropout_ffn(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(ff_output)
        tgt = self.norm2(tgt)
        
        if return_attention:
            return tgt, attn_weights
        return tgt

# --- Modified Multi-Modal Regression Transformer ---
class MultiModalRegressionTransformerWithWeights(nn.Module):
    """
    Modified version of the original model that can capture attention weights from specified layers
    """
    def __init__(self, original_model, layers_to_capture: Union[List[int], str] = "all"):
        super().__init__()
        # Copy all attributes from the original model
        self.hidden_dim = original_model.hidden_dim
        self.embeddings_module = original_model.embeddings_module
        self.regression_head = original_model.regression_head
        
        # Determine which layers to modify
        num_layers = len(original_model.transformer_decoder_layers)
        if layers_to_capture == "all":
            self.layers_to_capture = list(range(num_layers))
        elif isinstance(layers_to_capture, list):
            self.layers_to_capture = layers_to_capture
        else:
            raise ValueError("layers_to_capture must be 'all' or a list of layer indices")
        
        # Replace specified layers with modified versions
        self.transformer_decoder_layers = nn.ModuleList()
        
        for i, original_layer in enumerate(original_model.transformer_decoder_layers):
            if i in self.layers_to_capture:
                # Create modified layer that can return attention weights
                modified_layer = MaskedMultiHeadSelfAttentionBlockWithWeights(
                    d_model=original_layer.self_attn.embed_dim,
                    nhead=original_layer.self_attn.num_heads,
                    dim_feedforward=original_layer.linear1.out_features,
                    dropout=original_layer.dropout1.p
                )
                # Copy weights from original layer
                modified_layer.load_state_dict(original_layer.state_dict())
                self.transformer_decoder_layers.append(modified_layer)
            else:
                # Keep original layer unchanged
                self.transformer_decoder_layers.append(original_layer)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,
                token_type_vocab: dict,
                SMILES_fps: torch.Tensor,
                word_tokens_ref: torch.Tensor,
                values_ref: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                masked_lm_labels: torch.Tensor = None,
                capture_attention: bool = False):
        
        batch_size, sequence_length = token_type_ids.shape
        embeddings = self.embeddings_module(
            token_type_vocab=token_type_vocab,
            SMILES_fps=SMILES_fps,
            word_tokens_ref=word_tokens_ref,
            values_ref=values_ref,
            token_type_ids=token_type_ids
        )
        
        causal_mask = self.generate_square_subsequent_mask(sequence_length).to(embeddings.device)
        tgt_key_padding_mask = ~attention_mask
        transformer_output = embeddings
        
        attention_weights_by_layer = {}
        
        # Process each layer
        for i, layer in enumerate(self.transformer_decoder_layers):
            if i in self.layers_to_capture and capture_attention:
                transformer_output, attn_weights = layer(
                    tgt=transformer_output,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    return_attention=True
                )
                attention_weights_by_layer[f'layer_{i}'] = attn_weights
            else:
                transformer_output = layer(
                    tgt=transformer_output,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask
                )

        if masked_lm_labels is not None:
            masked_lm_positions = (masked_lm_labels != -100).nonzero(as_tuple=True)
            if masked_lm_positions[0].numel() > 0:
                masked_token_outputs = transformer_output[masked_lm_positions[0], masked_lm_positions[1]]
                predicted_values = self.regression_head(masked_token_outputs).squeeze(-1)
            else:
                predicted_values = torch.empty(0, device=embeddings.device)
            
            if capture_attention:
                return predicted_values, attention_weights_by_layer
            return predicted_values
        else:
            if capture_attention:
                return transformer_output, attention_weights_by_layer
            return transformer_output


def save_attention_weights(model, sample_input, save_dir="attention_weights", save_name="attention_weights"):
    """
    Capture and save attention weights from specified layers
    
    Args:
        model: The modified transformer model
        sample_input: Dictionary containing all input tensors
        save_dir: Directory to save the attention weights
        save_name: Base name for saved files
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass with attention capture
        output, attention_weights_dict = model(
            capture_attention=True,
            **sample_input
        )
    
    # Convert all attention weights to numpy
    attention_weights_np = {}
    for layer_name, weights in attention_weights_dict.items():
        attention_weights_np[layer_name] = weights.cpu().numpy()
    
    # Save as pickle file
    save_data = {
        'attention_weights': attention_weights_np,
        'input_shapes': {k: v.shape for k, v in sample_input.items() if isinstance(v, torch.Tensor)},
        'token_type_ids': sample_input['token_type_ids'].cpu().numpy(),
        'attention_mask': sample_input['attention_mask'].cpu().numpy(),
        'layers_captured': list(attention_weights_np.keys())
    }
    
    with open(os.path.join(save_dir, f"{save_name}.pkl"), 'wb') as f:
        pickle.dump(save_data, f)
    
    # Save individual layer weights as numpy files
    for layer_name, weights in attention_weights_np.items():
        np.save(os.path.join(save_dir, f"{save_name}_{layer_name}.npy"), weights)
    
    #print(f"Attention weights saved to {save_dir}/")
    print(f"Layers captured: {list(attention_weights_np.keys())}")
    #for layer_name, weights in attention_weights_np.items():
        #print(f"  {layer_name} shape: {weights.shape}")
        #print(f"    - Batch size: {weights.shape[0]}")
        #print(f"    - Number of heads: {weights.shape[1]}")
        #print(f"    - Sequence length: {weights.shape[2]} x {weights.shape[3]}")
    
    return attention_weights_np


def plot_attention_weights(attention_weights_dict, layer_name="layer_0", sample_idx=0, head_idx=0, 
                          token_type_ids=None, save_path=None, figsize=(12, 10)):
    """
    Plot attention weights as a heatmap for a specific layer
    
    Args:
        attention_weights_dict: Dictionary of attention weights by layer
        layer_name: Which layer to plot (e.g., "layer_0", "layer_1")
        sample_idx: Which sample from the batch to plot
        head_idx: Which attention head to plot
        token_type_ids: Optional token type information for labeling
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    if layer_name not in attention_weights_dict:
        print(f"Layer {layer_name} not found. Available layers: {list(attention_weights_dict.keys())}")
        return
    
    attention_weights = attention_weights_dict[layer_name]
    
    # Extract attention weights for specific sample and head
    attn_matrix = attention_weights[sample_idx, head_idx]
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(attn_matrix, 
                cmap='Reds', 
                cbar=True,
                square=True,
                cbar_kws={'label': 'Attention Weight'})
    
    plt.title(f'Attention Weights - {layer_name.title()}, Sample {sample_idx}, Head {head_idx}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Add token type information if available
    if token_type_ids is not None:
        seq_len = attn_matrix.shape[0]
        token_types = token_type_ids[sample_idx][:seq_len]
        
        # Add colored bars to show token types
        for i, token_type in enumerate(token_types):
            color = plt.cm.Set3(token_type % 12)
            plt.axhline(y=i+0.5, color=color, linewidth=2, alpha=0.7)
            plt.axvline(x=i+0.5, color=color, linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()





def hellow(hi):
    print(hi)




def create_modified_model_from_original(original_model, layers_to_capture: Union[List[int], str] = "all"):
    """
    Create a modified model that can capture attention weights from specified layers
    
    Args:
        original_model: Your original MultiModalRegressionTransformer
        layers_to_capture: Either "all" to capture all layers, or a list of layer indices [0, 1, 2, ...]
    """
    modified_model = MultiModalRegressionTransformerWithWeights(original_model, layers_to_capture)
    return modified_model


# Example usage function
def extract_and_save_attention_example():
    """
    Example of how to use the attention extraction functionality
    """
    # Assuming you have your original model loaded
    # original_model = MultiModalRegressionTransformer(...)
    # original_model.load_state_dict(torch.load('your_model.pth'))
    
    # Option 1: Capture attention from all layers
    # modified_model = create_modified_model_from_original(original_model, "all")
    
    # Option 2: Capture attention from specific layers only
    # modified_model = create_modified_model_from_original(original_model, [0, 2, 4])  # layers 0, 2, and 4
    
    # Option 3: Capture attention from just the first layer
    # modified_model = create_modified_model_from_original(original_model, [0])
    
    # Prepare sample input (you'll need to replace this with your actual data)
    sample_input = {
        'token_type_vocab': {'WORD_TOKEN': 0, 'SMILES_TOKEN': 1, 'VALUE_TOKEN': 2},
        'SMILES_fps': torch.randn(1, 10, 768),  # Replace with actual dimensions
        'word_tokens_ref': torch.randint(0, 100, (1, 10)),
        'values_ref': torch.randn(1, 10),
        'token_type_ids': torch.randint(0, 3, (1, 10)),
        'attention_mask': torch.ones(1, 10, dtype=torch.bool),
        'masked_lm_labels': None
    }
    
    # Extract and save attention weights
    # attention_weights_dict = save_attention_weights(modified_model, sample_input)
    
    # Plot attention weights from a specific layer
    # plot_attention_weights(attention_weights_dict, 
    #                       layer_name="layer_0",
    #                       token_type_ids=sample_input['token_type_ids'].numpy(),
    #                       save_path="attention_weights/layer_0_attention.png")
    
    # Compare attention patterns across layers
    # plot_attention_comparison(attention_weights_dict, 
    #                          layers_to_compare=["layer_0", "layer_1", "layer_2"],
    #                          token_type_ids=sample_input['token_type_ids'].numpy(),
    #                          save_path="attention_weights/layers_comparison.png")
    
    # Compare attention heads within a layer
    # plot_attention_heads_comparison(attention_weights_dict, 
    #                                layer_name="layer_0",
    #                                heads_to_compare=[0, 1, 2, 3],
    #                                save_path="attention_weights/heads_comparison.png")
    
    print("Example setup complete. Uncomment and modify the lines above with your actual model and data.")


if __name__ == "__main__":
    extract_and_save_attention_example()