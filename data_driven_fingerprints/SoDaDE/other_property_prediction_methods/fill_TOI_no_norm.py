from models import MultiTaskTaniamotoGP
from utils import train_test_split
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# also calculate the r2 score
from sklearn.metrics import mean_squared_error, r2_score

print("Starting script execution...") # Script start

# import the data
print("Loading canonical_smiles.csv...") # Before loading canonical_smiles.csv
df = pd.read_csv('other_property_prediction_methods/data.nosync/canonical_smiles.csv')
print(df.columns)
print("canonical_smiles.csv loaded.") # After loading canonical_smiles.csv

# import the fingerprints
print("Loading ecfp_fragprints.csv...") # Before loading ecfp_fragprints.csv
df_fingerprints = pd.read_csv('other_property_prediction_methods/data.nosync/features/ecfp_fragprints.csv', index_col=0)
print("ecfp_fragprints.csv loaded.") # After loading ecfp_fragprints.csv

def featurize(smiles):
    return df_fingerprints.loc[smiles]

cols = df.columns

# create a dataset for each task
dfs = []
feats = []
task_is = []
ys = []

print("Starting data featurization and data preparation loop...") # Before loop for data preparation
for col in cols:
    if col != 'Canonical Solvent SMILES':
        print(f"Processing column: {col}") # Inside loop, for each column
        # create a new dataframe with the smiles and the property
        df_temp = df[['Canonical Solvent SMILES', col]]
        # drop nas
        df_temp = df_temp.dropna()
        # add to the list
        dfs.append(df_temp)
        # featurize the data
        df_temp_feats = df_temp['Canonical Solvent SMILES'].apply(featurize).values
        feats.append(df_temp_feats)
        # generate the task index
        task_i = np.ones(len(df_temp_feats), dtype=int) * len(dfs)
        task_is.append(task_i)
        # get the target values
        y = df_temp[col].values
        # Append the pre-normalized y values directly
        ys.append(y)
print("Data featurization and data preparation loop complete.") # After loop for data preparation

print("Concatenating dataframes, features, task indices, and target values...") # Before concatenating data
# concatenate the dataframes
df = pd.concat(dfs, axis=0)
# concatenate the features
feats = np.concatenate(feats, axis=0)
# concatenate the task indices
task_is = np.concatenate(task_is, axis=0)
# concatenate the target values
ys = np.concatenate(ys, axis=0)
print("Concatenation complete.") # After concatenating data

print("Converting data to PyTorch tensors...") # Before converting to tensors
# split the data into train and test sets
x_train = torch.tensor(feats, dtype=torch.float64)
i_train = torch.tensor(task_is, dtype=torch.long) - 1
y_train = torch.tensor(ys, dtype=torch.float64)
print("Data converted to PyTorch tensors.") # After converting to tensors

print("Initializing MultiTaskTaniamotoGP model...") # Before model initialization
model = MultiTaskTaniamotoGP(
    n_tasks=12,
)
print("Model initialized.") # After model initialization

print("Starting model training...") # Before model training
# train the model
model.train((x_train, i_train), y_train)
print("Model training complete.") # After model training

# now make predictions on the table of interest
print("Loading table_of_interest_canonical_smiles.csv...") # Before loading table_of_interest_canonical_smiles.csv
df_table = pd.read_csv('other_property_prediction_methods/data.nosync/table_of_interest_canonical_smiles.csv')
print("table_of_interest_canonical_smiles.csv loaded.") # After loading table_of_interest_canonical_smiles.csv

print("Loading ecfp_fragprints_table_of_interest.csv...") # Before loading ecfp_fragprints_table_of_interest.csv
df_fingerprints_table = pd.read_csv('other_property_prediction_methods/data.nosync/features/ecfp_fragprints_table_of_interest.csv', index_col=0)
print("ecfp_fragprints_table_of_interest.csv loaded.") # After loading ecfp_fragprints_table_of_interest.csv

def featurize_table(smiles):
    return df_fingerprints_table.loc[smiles]

# create a dataset for each task
dfs_table = []
feats_table = []
task_is_table = []

print("Starting featurization loop for table of interest...") # Before loop for table of interest
for col in cols:
    if col != 'Canonical Solvent SMILES':
        print(f"Processing column for table of interest: {col}") # Inside loop for table of interest
        # create a new dataframe with the smiles and the property
        df_temp = df_table[['Canonical Solvent SMILES', col]]
        # add to the list
        dfs_table.append(df_temp)
        # featurize the data
        df_temp_feats = df_temp['Canonical Solvent SMILES'].apply(featurize_table).values
        feats_table.append(df_temp_feats)
        # generate the task index
        task_i = np.ones(len(df_temp_feats), dtype=int) * len(dfs_table)
        task_is_table.append(task_i)
print("Featurization loop for table of interest complete.") # After loop for table of interest

print("Concatenating features and task indices for table of interest...") # Before concatenating for table of interest
# concatenate the features
feats_table = np.concatenate(feats_table, axis=0)
# concatenate the task indices
task_is_table = np.concatenate(task_is_table, axis=0)
print("Concatenation for table of interest complete.") # After concatenating for table of interest

print("Creating prediction set tensors...") # Before creating prediction set tensors
# create the prediction set
x_test = torch.tensor(feats_table, dtype=torch.float64)
i_test = torch.tensor(task_is_table, dtype=torch.long) - 1
print("Prediction set tensors created.") # After creating prediction set tensors

print("Making predictions with the trained model...") # Before making predictions
# make predictions
y_pred = model.predict((x_test, i_test))
print("Predictions made.") # After making predictions

print("Assigning predictions to variables...") # Before assigning predictions
# The predictions are already in the correct scale, so just assign them.
y_pred_mean = y_pred[0].numpy()
y_pred_var = y_pred[1].numpy()
print("Predictions assigned.") # After assigning predictions

print("Creating dataframes for predictions...") # Before creating prediction dataframes
# create dataframe with the predictions, in the form of for each task in a column
col_data_mean = []
col_data_var = []
for i in range(len(dfs)): # use len(dfs) to get the number of tasks
    col_data_mean.append(y_pred_mean[i_test == i])
    col_data_var.append(y_pred_var[i_test == i])

# create a dataframe with the predictions, and correct the column names
df_table_pred_mean = pd.DataFrame(col_data_mean).T
df_table_pred_var = pd.DataFrame(col_data_var).T
df_table_pred_mean.columns = [f'Predicted {col}' for col in cols[:-1]]
df_table_pred_var.columns = [f'Predicted {col} variance' for col in cols[:-1]]

# add the smiles
df_table_pred_mean['Canonical Solvent SMILES'] = df_table['Canonical Solvent SMILES']
df_table_pred_var['Canonical Solvent SMILES'] = df_table['Canonical Solvent SMILES']
print("Prediction dataframes created.") # After creating prediction dataframes

print("Saving predictions to CSV files...") # Before saving predictions
# save the predictions
df_table_pred_mean.to_csv('other_property_prediction_methods/data.nosync/predictions_mean.csv', index=False)
df_table_pred_var.to_csv('other_property_prediction_methods/data.nosync/predictions_var.csv', index=False)
print("Predictions saved to predictions_mean.csv and predictions_var.csv.") # After saving predictions

print("Script execution finished.") # Script end