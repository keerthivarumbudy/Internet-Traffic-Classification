import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import math
import mutual_info_classif_preprocessing as pp
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB, MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

#converting results in string to intergers
y_str= list(set([x for x in pp.y]))
y_dict = dict()
for i,v in enumerate(y_str):
    y_dict[v]=i
pp.y = [y_dict[i] for i in pp.y]

#splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(pp.x,pp.y, test_size = 0.2)
y_test=pd.DataFrame(y_test)

#creating a Gaussian Classifier
gnb = GaussianNB()

#training the model using the training sets
res=gnb.fit(x_train, y_train)

#printing the score
print(gnb.score(x_test, y_test))
