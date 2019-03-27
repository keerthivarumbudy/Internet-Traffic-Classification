import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn.feature_selection import SelectKBest , mutual_info_classif
import sklearn as sk
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import math

from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('Finial_entry01_ITC_Dataset')
y = dataset.iloc[:,-1]
data = pd.read_csv('pearson_features(dataset2).csv')

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




x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
'''svc=SVC() # The default kernel used by SVC is the gaussian kernel
svc.fit(x_train, y_train)
prediction = svc.predict(x_test)
cm = confusion_matrix(y_test, prediction)
sum = 0
for i in range(cm.shape[0]):
    sum += cm[i][i]
accuracy = sum/x_test.shape[0]
print("a=",accuracy)'''

clf = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial').fit(x_train,y_train)
print(clf.score(x_test,y_test))
