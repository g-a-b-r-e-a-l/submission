import pandas as pd
import rdkit.Chem as Chem
from gauche.dataloader.molprop_loader import MolPropLoader

df_1 = pd.read_csv('SoDaDE/datasets/train_values.csv')
df_2 = pd.read_csv('SoDaDE/datasets/val_values.csv')

df = pd.concat([df_1, df_2])
print(f"Loaded {len(df)} rows from full_extracted_table.csv")

def canonicalize_smiles(smiles):
    """
    Convert a SMILES string to its canonical form.
    Handles non-string inputs gracefully.
    """
    # Ensure input is a string and not empty after stripping whitespace
    if not isinstance(smiles, str) or not smiles.strip():
        print(f"Skipping non-string or empty SMILES: {smiles}") # Optional: for debugging
        return None

    # Proceed with canonicalization for valid string SMILES
    mol = Chem.MolFromSmiles(smiles.strip()) # .strip() removes leading/trailing whitespace
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        print(f"Invalid SMILES string that could not be parsed by RDKit: {smiles}")
        return None

# apply the canonicalization function to the 'SMILES' column
df['Canonical Solvent SMILES'] = df['SMILES'].apply(canonicalize_smiles)
print(df['Canonical Solvent SMILES'][:5])
# drop the solvent and solvent smiles column
df.drop(columns=['solvent', 'SMILES'], inplace=True)

# save the df
df.to_csv('other_property_prediction_methods/data.nosync/canonical_smiles.csv', index=False)

# now load the data into the MolPropLoader
loader = MolPropLoader()
loader.read_csv('other_property_prediction_methods/data.nosync/canonical_smiles.csv', smiles_column='Canonical Solvent SMILES', label_column='N_mol_cm3')
# featurize the data
loader.featurize('ecfp_fragprints')

# create a df of the featurized data with the smiles as the index
df_featurized = pd.DataFrame(loader.features, index=df['Canonical Solvent SMILES'])

# save the featurized data
df_featurized.to_csv('other_property_prediction_methods/data.nosync/features/ecfp_fragprints.csv')

# now do the same for the table of interest
df_table = pd.read_csv('SoDaDE/datasets/test_values.csv')
# apply the canonicalization function to the 'SMILES' column

df_table['Canonical Solvent SMILES'] = df_table['SMILES'].apply(canonicalize_smiles)

# drop the solvent smiles column
df_table.drop(columns=['solvent','SMILES'], inplace=True)
# add a dummy y variable
df_table['y'] = 0

# save the df
df_table.to_csv('other_property_prediction_methods/data.nosync/table_of_interest_canonical_smiles.csv', index=False)



# now load the data into the MolPropLoader
loader = MolPropLoader()
loader.read_csv('other_property_prediction_methods/data.nosync/table_of_interest_canonical_smiles.csv', smiles_column='Canonical Solvent SMILES', label_column='y')
# featurize the data
loader.featurize('ecfp_fragprints')
# create a df of the featurized data with the smiles as the index
df_table_featurized = pd.DataFrame(loader.features, index=df_table['Canonical Solvent SMILES'])
# save the featurized data
df_table_featurized.to_csv('other_property_prediction_methods/data.nosync/features/ecfp_fragprints_table_of_interest.csv')