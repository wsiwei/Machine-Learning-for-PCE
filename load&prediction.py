import numpy as np
import pandas as pd
import os
import os.path
import matplotlib.pyplot as plt
from sklearn.metrics import *
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import joblib

model1 = joblib.load('RF_cir.model')

mol1 = Chem.MolFromSmiles('CC[N+]1=C(/C=C/C2=C(Cl)/C(=C/C3=C4Oc5ccccc5C=C4CCC3)CCC2)c2cccc3cccc1c23') #46.6
# mol2 = Chem.MolFromMolFile('./IRLy.sdf') #59
#
def get_labels():
    df = pd.read_excel('./molecules.xlsx')
    labels = []
    labels.append(pd.DataFrame(df['PCE(%)']))
    labels[0].dropna(axis=0, how='any', inplace=True)
    labels[0].reset_index(drop=True, inplace=True)
    # labels = ms_labels.fit_transform(labels[0])
    labels = np.array(labels[0])
    labels = np.array(labels).ravel()
    return labels
# #
circular_fingerprint = np.array(pd.read_csv('circular_fingerprint.csv'))
y = get_labels()
#
model = RandomForestRegressor(random_state=33) #at 1
model.fit(circular_fingerprint,y)
model = SelectFromModel(model,threshold=-np.inf,max_features=159, prefit=True)
#
# #
fp1 = Chem.GetMorganFingerprintAsBitVect(mol1, 2,nBits=1024)
# # fp2 = Chem.GetMorganFingerprintAsBitVect(mol2, 2,nBits=1024)
Efp1 = np.zeros((1,))
# # Efp2 = np.zeros((1,))
DataStructs.ConvertToNumpyArray( fp1 , Efp1)
# DataStructs.ConvertToNumpyArray( fp2 , Efp2)
Efp1 = Efp1.reshape(1,-1)
# # Efp2 = Efp2.reshape(1,-1)
Efp1 = model.transform(Efp1)
# # Efp2 = model1.transform(Efp2)
# #
result1 = model1.predict(Efp1)
# # result2 = model1.predict(Efp2)
# #
print(result1)
# # print(result2)