import torch

def create_collate_fn(token_type_vocab, masking_probability=0.15):
    """
    Returns a collate_fn that closes over token_type_vocab and masking_probability.
    """
    def collate_fn_with_args(batch_list_of_dicts):
        # Stack inputs from batch
        values_ref = torch.stack([d['values_tensor'] for d in batch_list_of_dicts])
        missing_val_mask = torch.stack([d['missing_val_mask'] for d in batch_list_of_dicts])
        word_tokens_ref = torch.stack([d['word_index_tensor'] for d in batch_list_of_dicts])
        token_type_ids = torch.stack([d['token_type_tensor'] for d in batch_list_of_dicts])
        SMILES_fps = torch.stack([d['chemberta_fps_tensor'] for d in batch_list_of_dicts])

        # 1. Initialize attention mask to all True (all tokens are initially "active")
        attention_mask = torch.ones(token_type_ids.shape, dtype=torch.bool)

        # 2. Apply the 'missing_val_mask' to the attention_mask.
        # These positions should be ignored by attention as they contain no information.
        # The 'token_type_ids' for these positions remain unchanged (e.g., VALUE_TOKEN).
        attention_mask[missing_val_mask] = False
        # 3. Initialize masked_lm_labels to ignore index (-100.0)
        # This is a common convention for PyTorch's CrossEntropyLoss to ignore loss at these positions.
        masked_lm_labels = torch.full(token_type_ids.shape, -100.0, dtype=torch.float)

        # --- Masking for training (MLM-style masking of *present* values) ---
        # Identify positions that are VALUE_TOKENs AND are NOT missing (i.e., contain actual numbers).
        # We only want to randomly mask existing, valid data points for the MLM objective.
        present_values_at_value_positions = (token_type_ids == token_type_vocab['VALUE_TOKEN']) & (~missing_val_mask)

        # Create a random mask for these selected positions based on masking_probability
        rand_tensor = torch.rand(values_ref.shape, device=values_ref.device)
        mlm_mask = (present_values_at_value_positions) & (rand_tensor < masking_probability)

        # Store the original values for the MLM-masked positions in masked_lm_labels.
        # These are the targets the model will try to predict.
        masked_lm_labels[mlm_mask] = values_ref[mlm_mask]

        # Change the token_type_id for the MLM-masked positions to MASK_TOKEN.
        # These tokens are part of the input sequence the model sees and attends to.
        token_type_ids[mlm_mask] = token_type_vocab['MASK_TOKEN']
        token_type_ids[missing_val_mask] = token_type_vocab['MASK_TOKEN']  # Ensure missing values retain their type

        # NOTE: The original `token_type_ids` for positions marked by `missing_val_mask`
        # are intentionally left as they were (e.g., `VALUE_TOKEN` if they were conceptually
        # value positions that were just NaN). Their exclusion from processing is handled
        # by setting `attention_mask` to `False` for those positions.

        return {
            'SMILES_fps' : SMILES_fps,
            'word_tokens_ref' : word_tokens_ref,
            'values_ref' : values_ref, # Contains original values, including NaNs where applicable.
            'token_type_ids': token_type_ids, # Contains MASK_TOKEN for MLM-masked positions, original types otherwise.
            'attention_mask': attention_mask, # False for original NaNs, True for all other active tokens (including MLM-masked).
            'masked_lm_labels': masked_lm_labels # Original values for MLM-masked positions, -100.0 for all others.
        }
    return collate_fn_with_args