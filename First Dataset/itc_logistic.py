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
from sklearn.metrics import precision_recall_fscore_support
'''from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score'''

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


dataset = pd.read_csv('Final_Derived_Sampe_ITC_Dataset')
y = dataset.iloc[:,-1]
data = pd.read_csv('pearson_features.csv')
x = data.values.astype(int)
y = y

k = 19
test = SelectKBest(mutual_info_classif, k)
features = test.fit_transform(x,y)
x_selected = features[:,0:k+1]
x_train, x_test, y_train, y_test = train_test_split(x_selected,y, test_size = 0.3)
class_names = set(y_test)
clf = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial').fit(x_train,y_train)
y_pred = clf.predict(x_test)
accuracy = (clf.score(x_test,y_test))
print(accuracy)
'''print(confusion_matrix(y_test,y_pred))
plot_confusion_matrix(y_test, clf.predict(x_test), classes=class_names,title='Confusion matrix, without normalization')
plt.show()'''
#print("precision, recall, F score",precision_recall_fscore_support(y_test, y_pred, average='weighted'))
