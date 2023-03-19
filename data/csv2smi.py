from rdkit import Chem
import pandas as pd

# read CSV file
df = pd.read_csv('/content/Quantum-GAN-implementation-using-PennyLane-and-IBMQ/data/qm9_5k.csv')

# extract SMILES strings from 'smiles' column
smiles_list = df['smiles'].tolist()

# convert SMILES strings to RDKit molecules
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

# write molecules to SMI file
with open('qm9_5k.smi', 'w') as f:
    for mol in mols:
        f.write(Chem.MolToSmiles(mol) + '\n')
