import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn.feature_selection import SelectKBest , mutual_info_classif
import sklearn as sk

dataset = pd.read_csv('Final_Derived_Sampe_ITC_Dataset')
y = dataset.iloc[:,-1]
data = pd.read_csv('pearson_features.csv')

#print(data)
#scaling up
'''for i in range(1,data.shape[1]):
    if data.ix[:,i].mean()< 1:
        data.ix[:,i]*=100000'''

x = data.values.astype(int)
y = y
test = SelectKBest(mutual_info_classif, k=19)
features = test.fit_transform(x,y)
x = features[:,0:20]
print("features selected")
