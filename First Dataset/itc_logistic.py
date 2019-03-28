import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import math
#import mutual_info_classif_preprocessing as pp
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest , mutual_info_classif

dataset = pd.read_csv('Final_Derived_Sampe_ITC_Dataset')
y = dataset.iloc[:,-1]
data = pd.read_csv('pearson_features.csv')
plt.xlabel = 'Number of features'
plt.ylabel = 'Accuracy'
x = data.values.astype(int)
y = y

#print(data)
#scaling up
'''for i in range(1,data.shape[1]):
    if data.ix[:,i].mean()< 1:
        data.ix[:,i]*=100000'''
#for i in range(1,121):
k = 21
test = SelectKBest(mutual_info_classif, k)
features = test.fit_transform(x,y)
x_selected = features[:,0:k]
x_train, x_test, y_train, y_test = train_test_split(x_selected,y, test_size = 0.3)
clf = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial').fit(x_train,y_train)
accuracy = (clf.score(x_test,y_test))
print(accuracy)
