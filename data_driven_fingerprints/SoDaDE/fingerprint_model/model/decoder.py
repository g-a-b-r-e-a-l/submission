import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# --- Positional Encoding Module ---
class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encodings to the input embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # Shape (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Register as buffer so it's not a model parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (embedded sequence).
                               Shape: (batch_size, sequence_length, d_model)
        Returns:
            torch.Tensor: Input tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- Masked Multi-Head Self-Attention Block ---
class MaskedMultiHeadSelfAttentionBlock(nn.Module):
    """
    A single block for a decoder-only Transformer model,
    implementing masked multi-head self-attention and a feed-forward network.
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

    def forward(self, tgt: torch.Tensor, tgt_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        attn_output, _ = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            is_causal=False
        )
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        ff_output = self.linear2(self.dropout_ffn(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(ff_output)
        tgt = self.norm2(tgt)
        return tgt

# --- FeedForward Neural Network ---
class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(FeedForwardNeuralNetwork, self).__init__()
        ff_dimension = 4 * input_dim
        self.fc1 = nn.Linear(input_dim, ff_dimension)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_dimension, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))

# --- Multi-Modal Input Embeddings ---
class MultiModalInputEmbeddings(nn.Module):
    def __init__(self, chemberta_fp_dim: int, column_vocab_size: int,
                 transformer_hidden_dim: int, max_sequence_length: int,
                 token_type_vocab_size: int, dropout_rate: float):
        super().__init__()
        self.transformer_hidden_dim = transformer_hidden_dim
        self.smiles_proj = FeedForwardNeuralNetwork(chemberta_fp_dim, transformer_hidden_dim)
        self.property_embedding = nn.Embedding(column_vocab_size, transformer_hidden_dim)
        self.value_proj = nn.Linear(1, transformer_hidden_dim)
        self.token_type_embeddings = nn.Embedding(token_type_vocab_size, transformer_hidden_dim)
        self.position_encodings = PositionalEncoding(transformer_hidden_dim, dropout_rate, max_sequence_length)
        self.LayerNorm = nn.LayerNorm(transformer_hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self,
                token_type_vocab: dict,
                SMILES_fps: torch.Tensor,
                word_tokens_ref: torch.Tensor,
                values_ref: torch.Tensor,
                token_type_ids: torch.Tensor,
                ):
        batch_size, max_batch_seq_len = token_type_ids.shape
        input_embeddings = torch.zeros(batch_size, max_batch_seq_len, self.transformer_hidden_dim,
                                       dtype=torch.float, device=token_type_ids.device)

        word_mask = (token_type_ids == token_type_vocab['WORD_TOKEN'])
        smiles_mask = (token_type_ids == token_type_vocab['SMILES_TOKEN'])
        value_mask = (token_type_ids == token_type_vocab['VALUE_TOKEN'])

        if word_mask.any():
            input_embeddings[word_mask] = self.property_embedding(word_tokens_ref[word_mask])
        if smiles_mask.any():
            input_embeddings[smiles_mask] = self.smiles_proj(SMILES_fps[smiles_mask])


        if value_mask.any():
            input_embeddings[value_mask] = self.value_proj(values_ref[value_mask].unsqueeze(-1))

        token_type_embedding_values = self.token_type_embeddings(token_type_ids)
        embeddings = input_embeddings + token_type_embedding_values
        embeddings = self.position_encodings(embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# --- Multi-Modal Regression Transformer ---
class MultiModalRegressionTransformer(nn.Module):
    def __init__(self, chemberta_fp_dim: int, column_vocab_size: int,
                 transformer_hidden_dim: int, max_sequence_length: int,
                 token_type_vocab_size: int, num_attention_heads: int,
                 num_transformer_layers: int, dropout_rate: float):
        super().__init__()
        self.hidden_dim = transformer_hidden_dim
        self.embeddings_module = MultiModalInputEmbeddings(
            chemberta_fp_dim=chemberta_fp_dim,
            column_vocab_size=column_vocab_size,
            transformer_hidden_dim=transformer_hidden_dim,
            max_sequence_length=max_sequence_length,
            token_type_vocab_size=token_type_vocab_size,
            dropout_rate=dropout_rate
        )
        self.transformer_decoder_layers = nn.ModuleList([
            MaskedMultiHeadSelfAttentionBlock(
                d_model=transformer_hidden_dim,
                nhead=num_attention_heads,
                dim_feedforward=transformer_hidden_dim * 4,
                dropout=dropout_rate
            )
            for _ in range(num_transformer_layers)
        ])
        self.regression_head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 4 * transformer_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(4 * transformer_hidden_dim),
            nn.Linear(4 * transformer_hidden_dim, 1)
        )
        self._init_weights()


    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,
                token_type_vocab: dict,
                SMILES_fps : torch.Tensor,
                word_tokens_ref : torch.Tensor,
                values_ref : torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                masked_lm_labels: torch.Tensor = None
                ) -> torch.Tensor:
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
        for layer in self.transformer_decoder_layers:
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
            return predicted_values
        else:
            return transformer_output

            

    def generative_inference(self,
                            token_type_vocab: dict,
                            SMILES_fps: torch.Tensor,
                            word_tokens_ref: torch.Tensor,
                            values_ref: torch.Tensor,
                            token_type_ids: torch.Tensor,
                            attention_mask: torch.Tensor,
                            positions_to_predict: list):
        """
        Generates missing values in a sequence.
        Returns only the predicted values at the specified positions.
        """
        
        # The values_ref tensor is used for context in the forward pass.
        # We don't need to modify it in a loop if we predict all positions at once.
        transformer_output = self.forward(
            token_type_vocab=token_type_vocab,
            SMILES_fps=SMILES_fps,
            word_tokens_ref=word_tokens_ref,
            values_ref=values_ref, # Pass the original tensor with nans/masked tokens
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        # Extract the hidden states for all positions we need to predict
        # Note: This assumes positions_to_predict is a list of integers
        hidden_states_to_predict = transformer_output[:, positions_to_predict, :]

        # The regression head can process all positions at once
        # Output shape will be (batch_size, num_positions_to_predict)
        predicted_values = self.regression_head(hidden_states_to_predict).squeeze(-1)

        return predicted_values
