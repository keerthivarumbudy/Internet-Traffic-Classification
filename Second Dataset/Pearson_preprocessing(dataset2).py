import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sklearn as sk
from sklearn.impute import SimpleImputer


data = pd.read_csv('Finial_entry01_ITC_Dataset',low_memory=False)

#y = data.iloc[:,248].values
#data = data.iloc[:,0:248].values
y = data.iloc[:,-1]
data = data.iloc[:,1:-1]
data.replace('?', np.nan, inplace=True)
data = data.iloc[:,0:248].values
print(data.shape[0])
print(data.shape[1])
imputer =SimpleImputer(missing_values=np.nan, strategy='mean')
imputer=imputer.fit(data)   
data=imputer.transform(data)
data=pd.DataFrame(data)


#for j in range(10):
#    for i in range(10):
        #if(data[i][j]=='?'):data[i][j]=0
#        print(data[i][j])
#data.to_csv('dataset2.csv')
#normalizing
'''x = data.values #returns a numpy array
min_max_scaler = sk.preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data = pd.DataFrame(x_scaled)
#scale up
for i in range(1,data.shape[1]):
    if data.ix[:,i].mean() < 0.001:
        data.ix[:,i]*=100000'''

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
print(data.shape)

data.to_csv('pearson_features(dataset2).csv')
'''
y_strings = list(set([x for x in y]))
y_dict = dict()
for i,v in enumerate(y_strings):
    y_dict[v]=i'''

#Processed Y
#Y = [y_dict[i] for i in y]
