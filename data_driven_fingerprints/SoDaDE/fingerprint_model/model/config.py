# --- 0. Model Configuration ---
DATA_PATH = "7376_train_dataset_norm.csv"
VAL_PATH = "560_val_dataset_norm.csv"





COLUMN_DICT = {
    'WORD_COLUMNS': ['solvent', 'Property_0', 'Property_1', 'Property_2',  'Property_3',  'Property_4', 'Property_5',
                        'Property_6', 'Property_7', 'Property_8', 'Property_9', 'Property_10',  'Property_11'],
    'VALUE_COLUMNS': ['Value_0', 'Value_1', 'Value_2', 'Value_3', 'Value_4', 'Value_5',
                        'Value_6','Value_7', 'Value_8', 'Value_9', 'Value_10', 'Value_11'],
    'SMILES_COLUMNS': 'SMILES'
}

TOKEN_TYPES = ['WORD_TOKEN', 'SMILES_TOKEN', 'VALUE_TOKEN', 'MASK_TOKEN', 'CLS_TOKEN', 'SEP_TOKEN']
WORD_TOKENS = ['alkane', 'aromatic', 'halohydrocarbon', 'ether', 'ketone', 'ester', 'nitrile', 'amine', 'amide', 'misc_N_compound', 'carboxylic_acid', 'monohydric_alcohol' , 'polyhydric_alcohol', 'other','ET30', 'alpha', 'beta', 'pi_star', 'SA', 'SB', 'SP', 'SdP', 'N_mol_cm3', 'n', 'fn', 'delta']
VOCAB_SIZE_COLUMNS = len(WORD_TOKENS)
TOKEN_TYPE_VOCAB_SIZE = len(TOKEN_TYPES)
token_type_vocab = {token_type: i for i, token_type in enumerate(TOKEN_TYPES)}
TOKEN_TYPE_VOCAB = token_type_vocab

TRANSFORMER_HIDDEN_DIM = 384
NUM_ATTENTION_HEADS = 8
NUM_TRANSFORMER_LAYERS = 3


MAX_SEQUENCE_LENGTH = 28
DROPOUT_RATE = 0.3
MASKING_PROBABILITY = 0.3

#--- 1. Training Configuration ---

NUM_EPOCHS = 40
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
