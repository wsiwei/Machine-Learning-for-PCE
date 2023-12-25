import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def read_excel(path):
  df=pd.read_excel(path,None)
  return df

def get_labels():
    df = read_excel('data.xlsx')
    labels = []
    labels.append(pd.DataFrame(df['molecules']['光热转换效率（%）']))
    labels[0].dropna(axis=0, how='any', inplace=True)
    labels[0].reset_index(drop=True, inplace=True)
    # labels = ms_labels.fit_transform(labels[0])

    labels = np.array(labels[0])
    labels = np.array(labels).ravel()
    return labels

s = get_labels()
s = pd.DataFrame(s)



plt.hist(s, bins=12, color=sns.desaturate("indianred", .8), alpha=.4)
plt.figure()
