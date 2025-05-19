import os

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem


def input_data_save(data_path: str, name: str, target: str, feature_column='canonical_smiles', vector_size=2048):
    base_path = fr"C:\Users\user\Desktop\madd_automl\data\{vector_size}_data"
    os.makedirs(base_path, exist_ok=True)

    df = pd.read_csv(data_path)
    df_x = df[feature_column].apply(lambda x: Chem.MolFromSmiles(x))
    df_x = df_x.dropna().apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, vector_size))

    X = np.array(df_x.tolist())
    X = pd.DataFrame(data=X)
    y = df[target]

    df_result = X.join(y)

    df_result.to_csv(os.path.join(base_path, f'{name}_{vector_size}.csv'), index=False)


if __name__ == "__main__":
    input_data_save(r'C:\Users\user\Desktop\madd_automl\data\500k_alz\alc_ds_500_processed.csv', 'alz_ds_500_final', target='docking_score', vector_size=1024)
