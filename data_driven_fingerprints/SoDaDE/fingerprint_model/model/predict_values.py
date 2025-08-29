from SoDaDE.fingerprint_model.model.config import (
    BATCH_SIZE,
    MAX_SEQUENCE_LENGTH,
    TOKEN_TYPE_VOCAB
)
import torch
from tqdm import tqdm

def predict_values(model, dataloader, optimizer, criterion, num_epochs, train=True, epoch=0):
    total_loss = 0
    num_batches_with_loss = 0  # Track batches that contributed to loss
    
    for batch_idx, batch_dict in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)):
        # 1. Move batch data to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in batch_dict.items()}
        
        if train:
            # 2. Zero the gradients
            optimizer.zero_grad()
        
        # 3. Forward pass
        predicted_values = model(TOKEN_TYPE_VOCAB,  **batch_dict)
        # 4. Get true masked labels
        true_masked_labels = inputs['masked_lm_labels'][inputs['masked_lm_labels'] != -100.0]
        
        # Check if there are any masked values in the current batch to avoid error
        if predicted_values.numel() > 0 and true_masked_labels.numel() > 0:
            loss = criterion(predicted_values, true_masked_labels)
            total_loss += loss.item()
            num_batches_with_loss += 1
            
            if train:
                # 5. Backward pass
                loss.backward()
                
                # 6. Update model parameters
                optimizer.step()
        else:
            # If no values were masked in this batch, or no valid labels, skip loss calculation
            print(f"Warning: No masked values for prediction in batch {batch_idx+1} of epoch {epoch+1}. Skipping loss calculation for this batch.")
    
    # Calculate average loss after processing all batches
    if num_batches_with_loss > 0:
        average_loss = total_loss / num_batches_with_loss
    else:
        average_loss = 0.0
        print("Warning: No batches had valid masked values for loss calculation.")
    
    return average_loss