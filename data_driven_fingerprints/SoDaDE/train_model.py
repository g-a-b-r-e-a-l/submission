import argparse
import textwrap
import pandas as pd
import os

from SoDaDE.fingerprint_model.model.dataset import load_dataset
from SoDaDE.fingerprint_model.model.collate import create_collate_fn
from SoDaDE.fingerprint_model.model.predict_values import predict_values

from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm # For a nice progress bar
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from SoDaDE.fingerprint_model.model.decoder import MultiModalRegressionTransformer

from SoDaDE.fingerprint_model.model.config import (VOCAB_SIZE_COLUMNS, TRANSFORMER_HIDDEN_DIM, 
                    MAX_SEQUENCE_LENGTH, TOKEN_TYPE_VOCAB_SIZE, 
                    NUM_ATTENTION_HEADS, NUM_TRANSFORMER_LAYERS,  
                    COLUMN_DICT, TOKEN_TYPE_VOCAB)


def train(
    number_of_epochs: int,
    learning_rate: float = 0.001,
    batch_size: int = 16,
    shuffle: bool = True,
    data_path: str = "7376_train_dataset_norm.csv",
    val_path: str = "560_val_dataset_norm.csv",
    masking_probability: float = 0.3,
    dropout_rate: float = 0.3,
    transformer_layers=3,
    model_dimension=384,
    attention_heads=8


):
    
    #Load the training and validation datasets
    dataset_train, chemberta_dimension = load_dataset(data_path, COLUMN_DICT, MAX_SEQUENCE_LENGTH)
    dataset_val, _ = load_dataset(val_path, COLUMN_DICT, MAX_SEQUENCE_LENGTH)

    #Wrap collate function to take additional variables
    configured_collate_fn = create_collate_fn(TOKEN_TYPE_VOCAB, masking_probability)


    # Create DataLoader for training and validation datasets
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, collate_fn=configured_collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle, collate_fn=configured_collate_fn)

    # Initialize the model
    model = MultiModalRegressionTransformer(
         chemberta_fp_dim=chemberta_dimension,
         column_vocab_size=VOCAB_SIZE_COLUMNS,
         transformer_hidden_dim=model_dimension,
         max_sequence_length=MAX_SEQUENCE_LENGTH,
         token_type_vocab_size=TOKEN_TYPE_VOCAB_SIZE,
         num_attention_heads=attention_heads,
         num_transformer_layers=transformer_layers,
         dropout_rate=dropout_rate
     )
    
    # Set up the optimizer and loss function and learning rate scheduler

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',         # 'min' for loss, 'max' for accuracy/F1 score
        factor=0.5,         # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=5,         # Number of epochs with no improvement after which learning rate will be reduced.
        #verbose=True,       # Print a message when LR is reduced
        min_lr=1e-8,        # Minimum learning rate to which it can be reduced
        cooldown=0          # Number of epochs to wait before resuming normal operation after lr has been reduced.
    )

    # Training loop
    best_val_loss = float('inf') # Initialize with a very large number
    train_loss_list = []
    val_loss_list = []
    model_folder = 'SoDaDE/fingerprint_model/saved_models_from_training'
# Create the directory if it doesn't exist
    os.makedirs(model_folder, exist_ok=True)

    for epoch in range(number_of_epochs):
        model.train() # Set the model to training mode
        # Use tqdm for a progress bar
        # 'desc' is the description, 'leave' keeps the bar after completion
        # 'position=0' helps if you have nested progress bars

        train_loss = predict_values(model, dataloader_train, optimizer, criterion, number_of_epochs, train=True, epoch=epoch)
        
        # 7. Validation step      
        model.eval()

        with torch.no_grad():
            val_loss = predict_values(model, dataloader_val, optimizer, criterion, number_of_epochs, train=False, epoch=epoch)
        
        # Update the learning rate scheduler
        scheduler.step(val_loss)

        print('train_loss = ', train_loss, 'val_loss = ', val_loss)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # Correct way to create a DataFrame from lists
        data = {
            'train_loss': train_loss_list,
            'val_loss': val_loss_list
        }
        new_df = pd.DataFrame(data)

        # It's good practice to add a .csv extension to the filename
        # Also, index=False prevents pandas from writing the DataFrame index as the first column in the CSV
        new_df.to_csv('SoDaDE/Loss_over_time.csv', index=False)

        print("Loss data saved to 'Loss_over_time.csv'")
        if val_loss < best_val_loss:
            print(f'Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...')
            best_val_loss = val_loss
            rounded = round(best_val_loss, 4)
            # Save the model's state_dict
            # Use os.path.join to create a file path that works on any operating system
            file_name = f'val_loss{rounded}_DPR_{dropout_rate}_MP_{masking_probability}_DM_{model_dimension}_TL_{transformer_layers}_heads_{attention_heads}.pth'
            best_model_path = os.path.join(model_folder, file_name)
            torch.save(model.state_dict(), best_model_path)
            
    return None


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train transformer.",
        epilog=textwrap.dedent(
            """To pass in arbitrary options, use the -c flag.
            Example usage:
                python train_model.py -learning_rate 0.001 -bs 32 -num_epochs 10
            """
        ),
    )
    argparser.add_argument("-num_epochs", "--number_of_epochs", type=int, help="Number of epochs to train the model.")
    argparser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    argparser.add_argument("-bs", "--batch_size", type=int, default=16, help="Batch size for training.")
    argparser.add_argument("-s", "--shuffle", type=bool, help="Shuffle the dataset before training.")
    argparser.add_argument("-td", "--train_data", type=str, help="Path to the training data file.")
    argparser.add_argument("-vd", "--val_data", type=str, help="Path to the validation data file.")
    argparser.add_argument("-mp", "--masking_probability", type=float, default=0.3, help="Probability of masking tokens in the input.")
    argparser.add_argument("-dr", "--dropout_rate", type=float, default=0.3, help="Dropout rate for the model.")
    argparser.add_argument("-tl", "--transformer_layers", type=int, default=3, help="Number of attention layers stacked")
    argparser.add_argument("-d_model", "--model_dimension", type=int, default=384, help="Model dimension, should be divisible by 8")
    argparser.add_argument("-at", "--attention_heads", type=int, default=8, help="Model dimension, should be divisible by 8")




    args = argparser.parse_args()

    results = train(
        number_of_epochs=args.number_of_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        data_path=args.train_data if args.train_data else "SoDaDE/fingerprint_model/datasets/train_set.csv",
        val_path=args.val_data if args.val_data else "SoDaDE/fingerprint_model/datasets/val_set.csv",
        masking_probability=args.masking_probability,
        dropout_rate=args.dropout_rate,
        transformer_layers=args.transformer_layers,
        model_dimension=args.model_dimension,
        attention_heads=args.attention_heads

    )
