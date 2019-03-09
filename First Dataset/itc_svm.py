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
import mutual_info_classif_preprocessing as pp
from sklearn.linear_model import LogisticRegression


x_train, x_test, y_train, y_test = train_test_split(pp.x,pp.y, test_size = 0.2)
'''svc=SVC() # The default kernel used by SVC is the gaussian kernel
svc.fit(x_train, y_train)
prediction = svc.predict(x_test)
cm = confusion_matrix(y_test, prediction)
sum = 0
for i in range(cm.shape[0]):
    sum += cm[i][i]

accuracy = sum/x_test.shape[0]
print(accuracy)'''

clf = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial').fit(x_train,y_train)
print(clf.score(x_test,y_test))
