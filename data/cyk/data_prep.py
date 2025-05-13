import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def get_fingerprint(df):
    df_x = df['Smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df_x = df_x.dropna().apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048))

    X = np.array(df_x.tolist())
    X = pd.DataFrame(data=X)
    y = df['pIC50']

    df_result = X.join(y)
    df_result.to_csv('cyk_fp_processed.csv', index=False)


if __name__ == "__main__":
    data_path = r"C:\Users\user\Desktop\madd_automl\data\cyk\cyk.csv"
    df_cyk = pd.read_csv(data_path)
    get_fingerprint(df_cyk)
