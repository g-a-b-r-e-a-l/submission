import pandas as pd
from torch.utils.data import Dataset
from SoDaDE.fingerprint_model.model.utils import (load_and_clean_csv, create_diff_token_vocabs,
                    create_values_tensor, word_token_indicies, 
                    create_token_type_tensor, create_fingerprints, 
                    load_chemberta_model_and_tokenizer)

from SoDaDE.fingerprint_model.model.config import WORD_TOKENS, TOKEN_TYPES

class MolecularPropertyDataset(Dataset):
    def __init__(self, df, max_seq_length, column_dict,
                 tokenizer, model_chemberta):
        self.df = df
        word_columns = column_dict['WORD_COLUMNS']
        value_columns = column_dict['VALUE_COLUMNS']
        smiles_column = column_dict['SMILES_COLUMNS']
        
        token_type_vocab, word_vocab = create_diff_token_vocabs(WORD_TOKENS, TOKEN_TYPES)

        self.values_tensor, self.missing_val_mask = create_values_tensor(df, max_seq_length, value_columns)
        self.word_index_tensor = word_token_indicies(df, max_seq_length, word_columns, word_vocab)
        self.token_type_tensor = create_token_type_tensor(df, max_seq_length, column_dict, token_type_vocab)

        self.chemberta_fps_tensor = create_fingerprints(df, max_seq_length, smiles_column, tokenizer, model_chemberta)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        batch_dict = {
            'values_tensor': self.values_tensor[idx],
            'missing_val_mask': self.missing_val_mask[idx],
            'word_index_tensor': self.word_index_tensor[idx],
            'token_type_tensor': self.token_type_tensor[idx],
            'chemberta_fps_tensor': self.chemberta_fps_tensor[idx]
        }

        return batch_dict


def load_dataset(data_path, column_dict, max_sequence_length=28):
    df = load_and_clean_csv(data_path)

    if df is None:
        raise ValueError("DataFrame is empty or not loaded correctly.")

    tokenizer, model_chemberta, chemberta_dimension = load_chemberta_model_and_tokenizer()

    dataset = MolecularPropertyDataset(df, 
                                       max_sequence_length, 
                                       column_dict, 
                                       tokenizer, 
                                       model_chemberta)
    
    return dataset, chemberta_dimension