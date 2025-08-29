import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


def load_and_clean_csv(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Data loaded from {DATA_PATH}. Shape: {df.shape}")
        return df

    except FileNotFoundError:
        print(f"Error: The file '{DATA_PATH}' was not found. Please ensure it's in the correct directory or update DATA_PATH.")

def load_chemberta_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        model_chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        model_chemberta.eval() # Set to evaluation mode for fingerprint generation
        chemberta_fp_dimension = model_chemberta.config.hidden_size
        print("ChemBERTa loaded successfully with hidden size:", chemberta_fp_dimension)
    except Exception as e:
        print(f"Error loading ChemBERta: {e}")
        exit()

    return tokenizer, model_chemberta, chemberta_fp_dimension

def create_values_tensor(df, max_seq_length, value_columns):
    values_tensor = torch.full((df.shape[0], max_seq_length), -100, dtype=torch.float)

    col_id = 1 # account for start token column
    for i in df.columns:
        if i in value_columns:
            values_tensor[:, col_id] = torch.tensor(df[i])
        col_id += 1

    missing_val_mask = torch.isnan(values_tensor)

    return values_tensor, missing_val_mask

def create_diff_token_vocabs(WORD_TOKENS, TOKEN_TYPES):
    word_vocab = {col: i for i, col in enumerate(WORD_TOKENS)}
    token_type_vocab = {token_type: i for i, token_type in enumerate(TOKEN_TYPES)}

    return token_type_vocab, word_vocab


def word_token_indicies(df, max_seq_length, word_columns, word_vocab):
    word_index_tensor = torch.zeros((df.shape[0], max_seq_length), dtype=torch.long)
    col_id = 1 # account for start token column

    for i in df.columns:
        if i in word_columns:
            word_index_tensor[:, col_id] = torch.tensor(df[i].map(word_vocab))
        col_id += 1
    return word_index_tensor

def create_fingerprints(df, max_seq_length, smiles_column, tokenizer, model_chemberta):
    smiles_list = df[smiles_column].tolist()
    chemberta_fp_dim = model_chemberta.config.hidden_size
    SMILES_fps = torch.zeros((df.shape[0], max_seq_length, chemberta_fp_dim), dtype=torch.float)

    # Tokenize SMILES strings
    smiles_tokenized_inputs = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        # Move inputs to device if you have a GPU for model_chemberta
        # smiles_tokenized_inputs = {k: v.to(device) for k, v in smiles_tokenized_inputs.items()}
        outputs = model_chemberta(**smiles_tokenized_inputs, output_hidden_states=True)
        smiles_chemberta_fps = outputs.hidden_states[-1].mean(dim=1) # Shape: (batch_size, CHEMBE`RTA_FP_DIM)

    SMILES_fps[:, 2, :] = smiles_chemberta_fps

    return SMILES_fps

def create_token_type_tensor(df, max_seq_length, column_dict, token_type_vocab):
    tensor_dict = {}

    for i in list(token_type_vocab.keys()):
        tensor_dict[i] = torch.full((df.shape[0], ), token_type_vocab[i], dtype=torch.int)

    token_type_tensor = torch.zeros(df.shape[0], max_seq_length, dtype=torch.int)

    token_type_tensor[:, 0] = tensor_dict['CLS_TOKEN']
    token_type_tensor[:, -1] = tensor_dict['SEP_TOKEN']
    tensor_index = 0
    for i in range(df.shape[1]):
        tensor_index = i+1 #starts at 1 becuase we have added a CLS column beforehand
        if df.columns[i] in column_dict['WORD_COLUMNS']:
            token_type_tensor[:, tensor_index] = tensor_dict['WORD_TOKEN']
        elif df.columns[i] in column_dict['VALUE_COLUMNS']:
            token_type_tensor[:, tensor_index] = tensor_dict['VALUE_TOKEN']
        elif df.columns[i] == column_dict['SMILES_COLUMNS']:
            token_type_tensor[:, tensor_index] = tensor_dict['SMILES_TOKEN']
        else:
            pass

    return token_type_tensor

