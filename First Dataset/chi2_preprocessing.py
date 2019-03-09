import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sklearn as sk

data = pd.read_csv('Final_Derived_Sampe_ITC_Dataset')

y = data.iloc[:,-1]
data = data.iloc[:,1:-1]
#print(data)
#normalizing
for i in range(1,data.shape[1]):
    if data.ix[:,i].mean()< 1:
        data.ix[:,i]*=100000
print(data)
