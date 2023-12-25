from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import DataStructs
import os
import os.path
import numpy as np
from rdkit import Chem
import pandas as pd

mols = []
smis = []
local = []

def get_data(path,list1):
    fileList = os.listdir(path)  # 获取path目录下所有文件
    for filename in fileList:
        pathTmp = os.path.join(path,filename) # 获取path与filename组合后的路径
        if os.path.isdir(pathTmp):  # 如果是目录
            get_data(pathTmp,list1) # 则递归查找
        elif filename[-4:].upper() == '.SDF':# 如果不是目录，则比较后缀名
            list1.append(pathTmp)
            try:
                sdf2mol(pathTmp)
            except OSError:
                osErorr.append(pathTmp)
            except TypeError:
                typeErorr.append(pathTmp)

def sdf2mol(path):
    mol = Chem.MolFromMolFile(path)
    mols.append(mol)
    smis.append(Chem.MolToSmiles(mol))

path = './'
get_data(path,local)

import numpy as np
import urllib.request
import pandas as pd
from rdkit import rdBase, Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from matplotlib import pyplot as plt

from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
def simi(fps):
    sm = [0]*44
    sm1 = []
    sm2 = []
    for i in range(len(fps)):
        sm[i] = 0
        for j in range(len(fps)):
            cons = DataStructs.FingerprintSimilarity(fps[i],fps[j])
            if cons > sm[i]:
                if cons == 1:
                    pass
                else:
                    sm[i] = cons
                    sm1.append(i)
                    sm2.append(j)
    sm_mean = np.mean(sm)
    return sm,sm_mean,sm1,sm2

fps1 = [Chem.GetHashedAtomPairFingerprintAsBitVect(x) for x in mols]
fps2 = [Chem.RDKFingerprint(x) for x in mols]
fps3 = [Chem.GetMorganFingerprintAsBitVect(x,2,nBits = 1024) for x in mols]

Chem.MolFromSmiles()


sm1,sm_mean1,af_num1,af_num2 = simi(fps1)
sm2,sm_mean2,tpf_num1,tpf_num2 = simi(fps2)
sm3,sm_mean3,mf_num1,mf_num2 = simi(fps3)

import seaborn as sns
import matplotlib.pyplot as plt

display = pd.DataFrame(columns=('Morgan','Topological','Atom_pair'))
display['Atom_pair'] = sm1
display['Topological'] = sm2
display['Morgan'] = sm3
# plt.boxplot(display,patch_artist=True,showmeans=True,boxprops = {'color':'black','facecolor':'#9999ff'},flierprops = {'marker':'o','markerfacecolor':'red','color':'black'},meanprops = {'marker':'D','markerfacecolor':'indianred'})
sns.swarmplot(data = display,orient = "h",size = 3,palette = sns.color_palette("dark",1,0.01))
axx = sns.boxplot(data = display,orient = "h",fliersize = 0.01)
axx.figure.savefig("Tanimoto_Coefficient.jpg",dpi = 300,bbox_inches='tight' )
# print(display)