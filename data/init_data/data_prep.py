import os

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem


def input_data_save_ic50(data_path: str, name: str, target: str, feature_column='canonical_smiles', vector_size=2048):
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
    data_paths = [
        # r"C:\Users\user\Desktop\madd_automl\data\init_data\alz.csv",
        # r"C:\Users\user\Desktop\madd_automl\data\init_data\skl.csv",
        r"C:\Users\user\Desktop\madd_automl\data\init_data\cancer_clear.csv",
        r"C:\Users\user\Desktop\madd_automl\data\init_data\dislip.csv",
        r"C:\Users\user\Desktop\madd_automl\data\init_data\parkinson.csv",
        r"C:\Users\user\Desktop\madd_automl\data\init_data\resistance.csv",
    ]
    for path in data_paths:
        name = Path(path).stem
        # input_data_save_ic50(path, name, target='IC50', vector_size=2048)
        input_data_save_ic50(path, name, target='IC50', vector_size=512)
