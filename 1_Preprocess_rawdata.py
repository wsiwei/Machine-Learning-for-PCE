import os
import os.path
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.model_selection import KFold
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import DataStructs
from sklearn.preprocessing import StandardScaler

def PCA_fp(train_x):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=127) #22
    reduce_X = pca.fit_transform(train_x)
    return reduce_X

def PCA_dp(train_x):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=108) #22
    reduce_X = pca.fit_transform(train_x)
    return reduce_X

ms_arr = StandardScaler()

smis = []
list1 = []
molecular_descriptors = []
circular_fingerprint = []
daylight_fingerprint = []
atompair_fingerprint = []
typeErorr = []
osErorr = []
bit_info =[]
Efps = []

def get_data(path,list1,arr1,arr2,arr3,arr4):
    fileList = os.listdir(path)  # get SDF file
    try:                                            # sort
        fileList.sort(key=lambda x:int(x))
    except:
        pass
    for filename in fileList:
        pathTmp = os.path.join(path,filename)
        if os.path.isdir(pathTmp):
            get_data(pathTmp,list1,arr1,arr2,arr3,arr4)
        elif filename[-4:].upper() == '.SDF':
            list1.append(pathTmp)
            try:
                sdf2mol(pathTmp)
            except OSError:
                osErorr.append(pathTmp)
            except TypeError:
                typeErorr.append(pathTmp)


def sdf2mol(path):
    mol = Chem.MolFromMolFile(path)
    # descriptors(mol)
    # cir_fp(mol)
    # tp_fp(mol)
    # at_fp(mol)
    smis.append(Chem.MolToSmiles(mol))

def descriptors(mol):
    descs = [desc_name[0] for desc_name in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    molecular_descriptors.append(desc_calc.CalcDescriptors(mol))


def cir_fp(mol):
    MOR_bitinfo = {}
    fp = np.zeros((1,))
    Efp = Chem.GetMorganFingerprintAsBitVect(mol, 2,bitInfo=MOR_bitinfo,nBits=1024)
    DataStructs.ConvertToNumpyArray( Efp , fp)
    bit_info.append(MOR_bitinfo)
    circular_fingerprint.append(fp)
    Efps.append(Efp)

def tp_fp(mol):     #日光
    DL_bitinfo = {}
    fp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(Chem.RDKFingerprint (mol,2,bitInfo=DL_bitinfo,fpSize = 1024), fp)
    # bit_info.append(DL_bitinfo)
    daylight_fingerprint.append(fp)

def at_fp(mol):
    fp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(Chem.GetHashedAtomPairFingerprintAsBitVect(mol,nBits = 1024), fp)
    atompair_fingerprint.append(fp)

def generate(arr1,arr2,arr3,arr4):
    arr1 = pd.DataFrame(arr1)
    arr1 = arr1.replace(np.inf, 0)
    arr1 = arr1.replace(np.nan, 0)
    # arr1 = PCA_dp(arr1)
    arr1 = ms_arr.fit_transform(np.array(arr1))
    arr1 = pd.DataFrame(arr1)
    descs = [desc_name[0] for desc_name in Descriptors._descList]
    arr1.columns = descs
    # pd.DataFrame(descs).to_csv(os.path.join(path, 'descriptors_name.csv'), index=False)
    # arr1.to_csv(os.path.join(path, 'descriptors.csv'), index=False)
    # ecfp4_names = [f'Morgan {i}' for i in range(len(arr2[0]))]
    # arr2 = pd.DataFrame(arr2,columns=ecfp4_names)
    # arr2.to_csv(os.path.join(path, 'circular_fingerprint.csv'), index=False)
    # arr3 = pd.DataFrame(arr3)
    # arr3.to_csv(os.path.join(path, 'Daylight_fingerprint.csv'), index=False)
    # arr4 = pd.DataFrame(arr4)
    # arr4.to_csv(os.path.join(path, 'atompair_fingerprint.csv'), index=False)
    # arr1 = PCA_dp(np.array(arr1))
    # arr2 = PCA_fp(np.array(arr2))
    # arr3 = PCA_fp(np.array(arr3))
    # arr4 = PCA_fp(np.array(arr4))
    # arr1 = pd.DataFrame(arr1)
    # arr2 = pd.DataFrame(arr2)
    # arr3 = pd.DataFrame(arr3)
    # arr4 = pd.DataFrame(arr4)
    # arr5_dpcf = pd.concat([arr1,arr2],axis=1,ignore_index=True)
    # arr5_dpcf = arr5_dpcf.reset_index(drop=True)
    # arr5_dpcf.to_csv(os.path.join(path, 'descriptors_cf.csv'), index=False)
    # arr6_dpdf = pd.concat([arr1, arr3], axis=1,ignore_index=True)
    # arr6_dpdf = arr6_dpdf.reset_index(drop=True)
    # arr6_dpdf.to_csv(os.path.join(path, 'descriptors_df.csv'), index=False)
    # arr7_dpatf = pd.concat([arr1, arr4], axis=1,ignore_index=True)
    # arr7_dpatf = arr7_dpatf.reset_index(drop=True)
    # arr7_dpatf.to_csv(os.path.join(path, 'descriptors_atf.csv'), index=False)

path1 = '../data/'
# path2 = '../nami/'
get_data(path1,list1,molecular_descriptors,circular_fingerprint,daylight_fingerprint,atompair_fingerprint)
# get_data(path2,list1,molecular_descriptors,circular_fingerprint,daylight_fingerprint,atompair_fingerprint)
# generate(molecular_descriptors,circular_fingerprint,daylight_fingerprint,atompair_fingerprint)


# key = open('key.txt','w')
# key.write(str(bit_info))
# key.close()

smis = pd.DataFrame(smis)


df1 = pd.read_excel('./molecules.xlsx')
lable1 = pd.DataFrame(df1['光热转换效率（%）'])
# df2 = pd.read_excel('../nami/nano.xlsx')
# lable2 = pd.DataFrame(df2['光热转换效率（%）'])
# df = pd.concat([lable1, lable2])
lable1.columns = ['PCE']
df = lable1.reset_index(drop=True)

all = pd.concat([smis,df],axis=1)
all.columns = ['smiles','PCE']
all.to_csv(os.path.join(path1, 'SMILES.csv'), index=False)
