from sklearn.ensemble import RandomForestRegressor
from utils import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# also calculate the r2 score
from sklearn.metrics import mean_squared_error, r2_score

print("Script started: fit_random_forest.py") # Script start

percentage = 0.8
plot = False # Set to True to enable plotting during cross-validation

# --- Data Loading and Preparation for Training ---
print("Loading 'canonical_smiles.csv'...")
df = pd.read_csv('other_property_prediction_methods/data.nosync/canonical_smiles.csv')
print("'canonical_smiles.csv' loaded.")

print("Loading 'ecfp_fragprints.csv'...")
df_fingerprints = pd.read_csv('other_property_prediction_methods/data.nosync/features/ecfp_fragprints.csv', index_col=0)
print("'ecfp_fragprints.csv' loaded.")

def featurize(smiles):
    """
    Featurizes a SMILES string using pre-loaded fingerprints.
    """
    return df_fingerprints.loc[smiles]

cols = df.columns

# Initialize lists to store data for each task
dfs = []
feats = []
task_is = []
ys = []

print("Starting data featurization loop for training data...")
for col in cols:
    print(col)
    if col != 'Canonical Solvent SMILES':
        print(f"Processing column (task) for training: {col}")
        # Create a new dataframe with the smiles and the property
        df_temp = df[['Canonical Solvent SMILES', col]]
        # Drop rows with NaN values for the current property
        df_temp = df_temp.dropna()
        # Add to the list
        dfs.append(df_temp)
        # Featurize the data
        df_temp_feats = df_temp['Canonical Solvent SMILES'].apply(featurize).values
        feats.append(df_temp_feats)
        # Generate the task index (1-based, will be adjusted later if needed)
        task_i = np.ones(len(df_temp_feats), dtype=int) * len(dfs)
        task_is.append(task_i)
        # Get the target values
        y = df_temp[col].values
        ys.append(y)
print("Data featurization for training data complete.")

# Concatenate all prepared data into single arrays for the Random Forest
print("Concatenating features, task indices, and target values for Random Forest input...")
feats_concatenated = np.concatenate(feats, axis=0)
task_is_concatenated = np.concatenate(task_is, axis=0) - 1 # Adjust task indices to be 0-based
ys_concatenated = np.concatenate(ys, axis=0)

# For Random Forest, combine features and task indices into a single input array
x_train_original = np.concatenate((feats_concatenated, task_is_concatenated.reshape(-1, 1)), axis=1)
y_train_original = ys_concatenated
print("Concatenation complete. Original dataset prepared for Random Forest.")

# --- Cross-Validation Loop for Random Forest ---
mses_all = {}
r2s_all = {}

# Initialize dictionaries for storing results per task
i_unique_tasks = np.unique(task_is_concatenated)
for i in i_unique_tasks:
    mses_all[i] = []
    r2s_all[i] = []

print("\nStarting cross-validation (25 seeds) for Random Forest...")
for seeds in range(25):
    print(f"\n--- Running cross-validation for seed: {seeds} ---")

    # Split the data into train and test sets for this seed
    # Note: train_test_split here returns (x_train, y_train), (x_test, y_test)
    # as i_train is not passed to it. The task index is part of x_train/x_test.
    print(f"Splitting data for seed {seeds}...")
    (x_train_split, y_train_split), (x_test_split, y_test_split) = train_test_split(
        x_train_original, y_train_original, percentage=percentage, seed=seeds
    )
    print(f"Data split complete for seed {seeds}.")

    # Create and fit the Random Forest model
    print(f"Initializing and training Random Forest model for seed {seeds}...")
    model = RandomForestRegressor(n_estimators=250, random_state=42) # Using 250 estimators as in original
    model.fit(x_train_split, y_train_split)
    print(f"Random Forest model trained for seed {seeds}.")

    # Make predictions on the test set
    print(f"Making predictions on test set for seed {seeds}...")
    y_pred_split = model.predict(x_test_split)
    print(f"Predictions made for seed {seeds}.")

    # Extract task indices from the test set for evaluation
    i_test_split = x_test_split[:, -1].astype(int) # Task index is the last column

    mses = {}
    r2s = {}
    print(f"Calculating MSE and R2 scores for each task for seed {seeds}...")
    for i in i_unique_tasks:
        mask = i_test_split == i
        if np.sum(mask) > 0: # Ensure there are samples for this task in the test set
            if plot:
                plt.scatter(y_test_split[mask], y_pred_split[mask], label=f'Task {i}')
            
            # Calculate the mse and r2 score
            mse = mean_squared_error(y_test_split[mask], y_pred_split[mask])
            r2 = r2_score(y_test_split[mask], y_pred_split[mask])
            mses[i] = mse
            r2s[i] = r2
            print(f"Task {i}: MSE = {mse:.4f}, R2 = {r2:.4f}")
        else:
            print(f"Task {i}: No samples in test set for this seed.")
            mses[i] = np.nan # Assign NaN if no samples
            r2s[i] = np.nan # Assign NaN if no samples

    if plot:
        print(f"Plotting predictions for seed {seeds}...")
        plt.plot([y_test_split.min(), y_test_split.max()], [y_test_split.min(), y_test_split.max()], 'k--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.title(f'RF Predictions by Task (Seed {seeds})')
        plt.show()
        print(f"Plotting complete for seed {seeds}.")

    # Print the average mse and r2 score for the current seed
    avg_mse = np.nanmean(list(mses.values())) # Use nanmean to handle NaNs
    avg_r2 = np.nanmean(list(r2s.values()))   # Use nanmean to handle NaNs
    
    print(f'\nAverage R2 Score for seed {seeds}: {avg_r2:.4f}')
    print(f'Average Mean Squared Error for seed {seeds}: {avg_mse:.4f}')
    
    # Append the mse and r2 score to the overall lists
    print(f"Appending results for seed {seeds} to overall lists...")
    for i in i_unique_tasks:
        mses_all[i].append(mses[i])
        r2s_all[i].append(r2s[i])
    print(f"Results appended for seed {seeds}.")
    
    print(f"Finished cross-validation for seed {seeds}.")

print("\nAll cross-validation runs complete for Random Forest.")

# Save the cross-validation results
print("Saving cross-validation results to CSV files (mses_rf.csv, r2s_rf.csv)...")
df_mses = pd.DataFrame(mses_all)
df_mses.to_csv('other_property_prediction_methods/data.nosync/mses_rf.csv', index=False)
df_r2s = pd.DataFrame(r2s_all)
df_r2s.to_csv('other_property_prediction_methods/data.nosync/r2s_rf.csv', index=False)
print("Cross-validation results saved.")

# --- Train Final Random Forest Model on All Data for Prediction ---
print("\nTraining final Random Forest model on the entire dataset for predictions...")
final_rf_model = RandomForestRegressor(n_estimators=250, random_state=42)
final_rf_model.fit(x_train_original, y_train_original)
print("Final Random Forest model trained.")

# --- Make Predictions on the "Table of Interest" ---
print("\nLoading 'table_of_interest_canonical_smiles.csv' for prediction...")
df_table = pd.read_csv('other_property_prediction_methods/data.nosync/table_of_interest_canonical_smiles.csv')
print("'table_of_interest_canonical_smiles.csv' loaded.")

print("Loading 'ecfp_fragprints_table_of_interest.csv' for prediction...")
df_fingerprints_table = pd.read_csv('other_property_prediction_methods/data.nosync/features/ecfp_fragprints_table_of_interest.csv', index_col=0)
print("'ecfp_fragprints_table_of_interest.csv' loaded.")

def featurize_table(smiles):
    """
    Featurizes a SMILES string from the table of interest using its specific fingerprints.
    """
    return df_fingerprints_table.loc[smiles]

# Prepare data for prediction on the table of interest
dfs_table = []
feats_table = []
task_is_table = []

print("Starting featurization loop for table of interest...")
for col_idx, col in enumerate(cols): # Use enumerate to get a 0-based index for tasks
    if col != 'Canonical Solvent SMILES':
        print(f"Processing column for table of interest: {col}")
        df_temp = df_table[['Canonical Solvent SMILES', col]] # 'col' here is just for column names, not actual values
        
        df_temp_feats = df_temp['Canonical Solvent SMILES'].apply(featurize_table).values
        feats_table.append(df_temp_feats)
        
        # Assign the correct 0-based task index for prediction
        # The task index for prediction should correspond to the task index used during training
        # which was `len(dfs) - 1` (0-based) in the training loop.
        # Here, `col_idx - 1` gives the correct 0-based index if 'Canonical Solvent SMILES' is the first column.
        # Otherwise, it's just `col_idx` if we iterate only over property columns.
        # Given `cols` includes 'Canonical Solvent SMILES', `col_idx` for the first property will be 1.
        # So, `col_idx - 1` correctly maps to 0-based task indices.
        task_i = np.ones(len(df_temp_feats), dtype=int) * (col_idx) #removed -1
        task_is_table.append(task_i)
print("Featurization loop for table of interest complete.")

print("Concatenating features and task indices for table of interest...")
feats_table_concatenated = np.concatenate(feats_table, axis=0)
task_is_table_concatenated = np.concatenate(task_is_table, axis=0)

# Combine features and task indices for the Random Forest prediction input
x_predict_table = np.concatenate((feats_table_concatenated, task_is_table_concatenated.reshape(-1, 1)), axis=1)
print("Concatenation for table of interest complete. Prediction input prepared.")

print("Making predictions on the table of interest with the final Random Forest model...")
y_pred_table = final_rf_model.predict(x_predict_table)
print("Predictions made.")

print("Creating dataframe for predictions...")
# Create a list of prediction arrays, one for each task
col_data_mean = []
for i in i_unique_tasks:
    print('I = ', i)
    mask = task_is_table_concatenated == i
    if i == 12:
        print(mask)
    col_data_mean.append(y_pred_table[mask])

# Create a DataFrame with the predictions, with columns named after the original properties
# Skip 'Canonical Solvent SMILES' from original 'cols' when naming prediction columns
prediction_cols = [f'Predicted {col}' for col in cols if col != 'Canonical Solvent SMILES']
df_table_pred_mean = pd.DataFrame(col_data_mean).T
df_table_pred_mean.columns = prediction_cols

# Add the Canonical Solvent SMILES back to the prediction DataFrame
df_table_pred_mean['Canonical Solvent SMILES'] = df_table['Canonical Solvent SMILES']
print("Prediction dataframe created.")

print("Saving predictions to CSV file (predictions_rf_mean.csv)...")
df_table_pred_mean.to_csv('other_property_prediction_methods/data.nosync/predictions_rf_mean.csv', index=False)
print("Predictions saved to 'predictions_rf_mean.csv'.")

# Note: Random Forest does not directly provide prediction variances in the same way a Gaussian Process does.
# If uncertainty estimates are needed for Random Forest, methods like calculating the standard deviation
# of predictions from individual trees can be explored, but this is not a direct output of .predict().

print("Script execution finished: fit_random_forest.py")