import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sklearn as sk

data = pd.read_csv('Final_Derived_Sampe_ITC_Dataset')

y = data.iloc[:,-1]
data = data.iloc[:,1:-1]
print("1\n",data)
#normalizing
x = data.values #returns a numpy array
min_max_scaler = sk.preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data = pd.DataFrame(x_scaled)
print("2\n",data)
#scale up
for i in range(1,data.shape[1]):
    if data.ix[:,i].mean() < 0.001:
        data.ix[:,i]*=100000

#print("3\n",data)

###########
corr = data.corr()
#print(corr)
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        '''or np.isnan(corr.iloc[i,j])'''
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
        elif np.isnan(corr.iloc[i,j]):
            if columns[j]:
                columns[j] = False
                break

selected_columns = data.columns[columns]
data = data[selected_columns]
print(data.shape, data)

y_strings = list(set([x for x in y]))
y_dict = dict()
for i,v in enumerate(y_strings):
    y_dict[v]=i

#Processed Y
Y = [y_dict[i] for i in y]
