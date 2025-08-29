import torch

def create_fine_collate_fn(token_type_vocab):
    """
    Returns a collate_fn that closes over token_type_vocab and masking_probability.
    """
    def collate_fn_with_args(batch_list_of_dicts):
        values_ref_list = [d['values_tensor'] for d in batch_list_of_dicts]
        word_tokens_ref_list = [d['word_index_tensor'] for d in batch_list_of_dicts]
        token_type_ids_list = [d['token_type_tensor'] for d in batch_list_of_dicts]
        SMILES_fps_list = [d['chemberta_fps_tensor'] for d in batch_list_of_dicts]

        values_ref = torch.stack(values_ref_list)
        word_tokens_ref = torch.stack(word_tokens_ref_list)
        token_type_ids = torch.stack(token_type_ids_list)
        SMILES_fps = torch.stack(SMILES_fps_list)

        attention_mask = torch.ones(token_type_ids.shape, dtype=torch.bool)
        target_values = values_ref[:, 26].clone().detach()        
        values_ref[:, 26] = torch.nan

        missing_vals = torch.isnan(values_ref)

        token_type_ids[missing_vals] = token_type_vocab['MASK_TOKEN'] # Apply mask for existing missing values
        return {
            'SMILES_fps' : SMILES_fps,
            'word_tokens_ref' : word_tokens_ref,
            'values_ref' : values_ref,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'target_values': target_values
        }
    return collate_fn_with_args