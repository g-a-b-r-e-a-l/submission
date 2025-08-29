from models import MultiTaskTaniamotoGP
from utils import train_test_split
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# also calculate the r2 score
from sklearn.metrics import mean_squared_error, r2_score

print("Script started.") # Start of script

percentage = 0.8
plot = False

# import the data
print("Loading 'canonical_smiles.csv'...") # Before loading canonical_smiles.csv
df = pd.read_csv('other_property_prediction_methods/data.nosync/canonical_smiles.csv')
print("'canonical_smiles.csv' loaded.") # After loading canonical_smiles.csv

# import the fingerprints
print("Loading 'ecfp_fragprints.csv'...") # Before loading ecfp_fragprints.csv
df_fingerprints = pd.read_csv('other_property_prediction_methods/data.nosync/features/ecfp_fragprints.csv', index_col=0)
print("'ecfp_fragprints.csv' loaded.") # After loading ecfp_fragprints.csv

def featurize(smiles):
    return df_fingerprints.loc[smiles]

cols = df.columns

# create a dataset for each task
dfs = []
feats = []
task_is = []
ys = []

y_means = []
y_vars = []

print("Starting loop to process each column (task)...") # Before loop for data processing
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

        '''# normalize the target values
        y_mean = np.mean(y)
        y_var = np.var(y)

        y_means.append(y_mean)
        y_vars.append(y_var)
        y = (y - y_mean) / np.sqrt(y_var)'''
        ys.append(y)
print("Finished processing all columns (tasks).") # After loop for data processing

print("Concatenating features, task indices, and target values...") # Before concatenation
# concatenate the dataframes
df = pd.concat(dfs, axis=0)
# concatenate the features
feats = np.concatenate(feats, axis=0)
# concatenate the task indices
task_is = np.concatenate(task_is, axis=0)
# concatenate the target values
ys = np.concatenate(ys, axis=0)
print("Concatenation complete.") # After concatenation

print("Converting data to PyTorch tensors for original dataset...") # Before converting to tensors
# split the data into train and test sets
x_train_original = torch.tensor(feats, dtype=torch.float64)
i_train_original = torch.tensor(task_is, dtype=torch.long)
y_train_original = torch.tensor(ys, dtype=torch.float64)
print("Data converted to PyTorch tensors.") # After converting to tensors

mses_all = {}
r2s_all = {}

i_unique = np.unique(i_train_original)
for i in i_unique:
    mses_all[i] = []
    r2s_all[i] = []

print("Starting main loop for cross-validation (25 seeds)...") # Before the main cross-validation loop
for seeds in range(25):
    print(f"\n--- Running for seed: {seeds} ---") # At the beginning of each seed iteration
    # split the data into train and test sets
    print(f"Splitting data for seed {seeds}...") # Before data split
    (x_train, i_train, y_train), (x_test, i_test, y_test) = train_test_split(x_train_original, y_train_original, i_train_original, percentage=percentage, seed=seeds)
    print(f"Data split complete for seed {seeds}.") # After data split

    # create the model
    print(f"Initializing MultiTaskTaniamotoGP model for seed {seeds}...") # Before model creation
    model = MultiTaskTaniamotoGP(
        n_tasks=len(dfs) + 1,
    )
    print(f"Model initialized for seed {seeds}.") # After model creation

    # fit the model, try three times
    print(f"Attempting to train model for seed {seeds}...") # Before training attempts
    try:
        model.train((x_train, i_train), y_train)
        print(f"Model trained successfully on first attempt for seed {seeds}.") # After successful first attempt
    except Exception as e:
        print(f"First training attempt failed for seed {seeds}. Retrying...") # After first attempt failure
        try:
            model.train((x_train, i_train), y_train)
            print(f"Model trained successfully on second attempt for seed {seeds}.") # After successful second attempt
        except Exception as e:
            print(f"Second training attempt failed for seed {seeds}. Retrying...") # After second attempt failure
            try:
                model.train((x_train, i_train), y_train)
                print(f"Model trained successfully on third attempt for seed {seeds}.") # After successful third attempt
            except Exception as e:
                print(f"Failed to train the model after 3 attempts for seed {seeds}.") # After all attempts failed
                print(e)
                continue
    print(f"Model training process completed for seed {seeds}.") # After training attempts (success or failure)

    # make predictions
    print(f"Converting test data to PyTorch tensors for seed {seeds}...") # Before converting test data to tensors
    x_test = torch.tensor(x_test, dtype=torch.float64)
    i_test = torch.tensor(i_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.float64)
    print(f"Test data converted for seed {seeds}.") # After converting test data to tensors

    print(f"Making predictions for seed {seeds}...") # Before making predictions
    y_pred = model.predict((x_test, i_test))
    print(f"Predictions made for seed {seeds}.") # After making predictions

    mses = {}
    r2s = {}
    # plot the predictions by task
    i_unique = np.unique(i_test)
    print(f"Calculating MSE and R2 scores for each task for seed {seeds}...") # Before calculating metrics per task
    for i in i_unique:
        mask = i_test == i
        if plot:
            plt.scatter(y_test[mask], y_pred[0][mask], label=f'Task {i}')
        # calculate the mse and r2 score
        mse = mean_squared_error(y_test[mask], y_pred[0][mask])
        r2 = r2_score(y_test[mask], y_pred[0][mask])
        mses[i] = mse
        r2s[i] = r2
        print(f"Task {i} - MSE: {mse:.4f}, R2: {r2:.4f}") # Per task metric print
    print(f"Metrics calculated for all tasks for seed {seeds}.") # After calculating metrics per task

    if plot:
        print(f"Plotting predictions for seed {seeds}...") # Before plotting
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.title('Predictions by Task')
        plt.show()
        print(f"Plotting complete for seed {seeds}.") # After plotting

    # print the average mse and r2 score
    avg_mse = np.mean(list(mses.values()))
    avg_r2 = np.mean(list(r2s.values()))

    print(f'\nAverage R2 Score for seed {seeds}: {avg_r2:.4f}') # Average R2
    print(f'Average Mean Squared Error for seed {seeds}: {avg_mse:.4f}') # Average MSE
    
    if plot:
        print()
        # now print the mse and r2 score for each task
        for i in mses.keys():
            print(f'Task {i}:')
            print(f'Mean Squared Error: {mses[i]:.4f}')
            print(f'R2 Score: {r2s[i]:.4f}')
            print()
    
    # append the mse and r2 score to the list
    print(f"Appending results for seed {seeds} to overall lists...") # Before appending results
    for i in mses.keys():
        mses_all[i].append(mses[i])
        r2s_all[i].append(r2s[i])
    print(f"Results appended for seed {seeds}.") # After appending results
    
    print(f"Finished with seed {seeds}.") # End of each seed iteration

print("\nAll cross-validation runs complete.") # After the main cross-validation loop

print("Saving results to CSV files...") # Before saving results
# save the results as a df with columns as the task indices and rows as the seeds
df_mses = pd.DataFrame(mses_all)
df_mses.to_csv('other_property_prediction_methods/data.nosync/mses_multitask_gp.csv', index=False)
df_r2s = pd.DataFrame(r2s_all)
df_r2s.to_csv('other_property_prediction_methods/data.nosync/r2s_multitask_gp.csv', index=False)
print("Results saved to 'mses_multitask_gp.csv' and 'r2s_multitask_gp.csv'.") # After saving results

print("Script finished.") # End of script