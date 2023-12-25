
import os
import os.path
from rdkit.Chem import AllChem as Chem
import shap
import joblib
import pandas as pd
from matplotlib import cm
from sklearn import preprocessing
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
import matplotlib.pyplot as plt


from jupyterthemes import jtplot
jtplot.style(theme='default',grid = False)

key = open('key.txt','r')
a = key.read()
bit_info = eval(a)
key.close()
mols = []

# shap.initjs()

def get_mol(path):
    fileList = os.listdir(path)  # 获取path目录下所有文件
    for filename in fileList:
        pathTmp = os.path.join(path,filename) # 获取path与filename组合后的路径
        if os.path.isdir(pathTmp):  # 如果是目录
            get_mol(pathTmp) # 则递归查找
        elif filename[-4:].upper() == '.SDF':# 如果不是目录，则比较后缀名
            mols.append(Chem.MolFromMolFile(pathTmp))

path = './'
get_mol(path)

model1 = joblib.load(filename="RF_cir.model")
model2 = joblib.load(filename="svm_des.model")

# explainer = shap.TreeExplainer(model1)
df1 = pd.read_csv('circular_fingerprint_selected.csv')
df2 = pd.read_csv('descriptors_train.csv')
explainer_rf = shap.TreeExplainer(model1)
explainer_svm = shap.KernelExplainer(model2.predict,df2)

# df1 = pd.read_csv('train_x_cb.csv')
# name = pd.read_csv('descriptors_name.csv')
# df1.columns = name


shap_values = explainer_rf.shap_values(df1)
shap_values2 = explainer_svm.shap_values(df2)

# shap.summary_plot(shap_values2, df2,max_display=20,show=False)
# plt.savefig(('./svm_des_shap.png'),dpi = 600,bbox_inches = 'tight')
shap.summary_plot(shap_values, df1,max_display=20,show=False)
plt.savefig(('./RF_cir_shap.png'),dpi = 600,bbox_inches = 'tight')

aaa = []
for i in range(44):
    try:
        aaa.append(i)
        aaa.append(bit_info[i][12])
    except KeyError:
        pass

def get_info(mol_index, bit_ID, bitinfo, shap_info):
    info = bitinfo[mol_index][bit_ID]
    scaled = preprocessing.minmax_scale(shap_info[:,bit_ID])
    colour_info = [[cm.viridis(scaled[mol_index])]] * len(info)
    atom_info = []
    for i in info:
        atom_info.append(i[0])
    colour = dict(zip(atom_info, colour_info))

    radius_info = [info[0][1]] * len(info)
    radius = dict(zip(atom_info, radius_info))

    return colour, radius

def frag_shap(colour, radius, mol):
    d2d = rdMolDraw2D.MolDraw2DSVG(600,280,300,280)
    d2d.DrawMoleculeWithHighlights(mol,"98513984",colour,colour,radius,{})
    d2d.FinishDrawing()
    a = d2d.GetDrawingText()
    return SVG(a)


colour, radius = get_info(16,318,bit_info,shap_values2)
frag_shap(colour, radius,mols[16])


def singleV(nmb):
    import matplotlib.pyplot as plt
    shap_values = explainer_rf(df1)
    shap_values2 = explainer_rf(df2)
    shap_valuessss = explainer_rf.shap_values(df2.iloc[nmb])
    # shap.force_plot(explainer_rf.expected_value, shap_valuessss, df2.iloc[2])
    # shap.plots.waterfall(shap_valuessss)
    # shap.plots._waterfall.waterfall_legacy(explainer_rf.expected_value[0], shap_valuessss, df2.iloc[nmb])
    shap.plots._waterfall.waterfall_legacy(explainer_svm.expected_value, shap_valuessss, df2.iloc[nmb], show=False)
    plt.savefig(('./vis_img/%i/shap2.png' %nmb),dpi = 600,bbox_inches = 'tight')

import matplotlib.pyplot as plt
singleV(19)