'''
Supervised Learning Approach on CK+ dataset

# Libraries Used
    pandas, sklearn, numpy

# Variables
    clf_svm : SVM classifier
    clf_knn : KNN classifier
    clf_rf : Random Forest classifier
    clf_nb : Naive Bayes classifier
    clf_nn : Multi-Layer Perceptron classifier
    clf_ada : AdaBoost Classifier
    df : pandas dataframe
    df_labelled : Dataframe of labelled data
    df_unlabelled : Dataframe of unlabelled data
    X : inputs
    y : labels
    pred : predictions
'''

# Libraries used
import pandas as pd                                                             # To read database
from sklearn.cross_validation import train_test_split                           # To split database
from sklearn.metrics import accuracy_score, precision_recall_fscore_support     # Result of model on database
import numpy as np                                                              # Mathematical analysis
from sklearn.svm import SVC                                                     # Importing Support Vector Classifier
import matplotlib.pyplot as plt                                                 # Plotting data

# Initialization for graph
xplt=[0.01,0.05,0.1,0.5,1,2,4,8,10,15]
ratio=[0.1,0.2,0.3,0.4]
acc_plt=[[],[],[],[]]

def func(x,j):
    '''
    Takes as input C for SVM and split ratio and updates data
    '''
    clf_svm = SVC(kernel="rbf", C=x)

    # Dataframe
    df = pd.read_csv("emotion.csv",header=0)
    df = df.drop(["Person Id", "Person SubID"],axis=1)
    df_labelled = df[df["Emotion"]!=-1]
    df_unlabelled = df[df["Emotion"]==-1]

    # Labelled and unlabelled data
    # Expectation Maximization
    y_labelled = df_labelled["Emotion"]
    X_labelled = df_labelled.drop(["Emotion"],axis=1)
    y_unlabelled = df_unlabelled["Emotion"]
    X_unlabelled = df_unlabelled.drop(["Emotion"],axis=1)

    clf_svm.fit(X_labelled,y_labelled)
    pred = clf_svm.predict(X_unlabelled)
    df["Emotion"][df["Emotion"]==-1]=pred

    # Final database
    X=df.drop(["Emotion"],axis=1)
    y=df["Emotion"]

    # Split the data into training and validation dataset
    X_train, X_cv, y_train, y_cv = train_test_split(X,y,test_size=j,random_state=42)

    # Get Accuracy plot for data
    clf_svm = SVC(kernel="rbf", C=450000)
    clf_svm.fit(X_train, y_train)
    pred = clf_svm.predict(X_cv)
    a=accuracy_score(pred, y_cv)
    a=a.astype(np.float64)
    if j==0.1:
        i=0
    elif j==0.2:
        i=1
    elif j==0.3:
        i=2
    else:
        i=3
    acc_plt[i].append(float(a)*100)

# Get plots for data
for j in ratio:
    for i in xplt:
        func(i,j)

# Plot the data
plt.figure()
plt.plot(xplt,acc_plt[0],"navy",label="Split Ratio - 0.1")
plt.plot(xplt, acc_plt[1],"darkorange",label="Split Ratio - 0.2")
plt.plot(xplt,acc_plt[2],"red",label="Split Ratio - 0.3")
plt.plot(xplt,acc_plt[3],color="green",label="Split Ratio - 0.4")
plt.xlabel('Value of C')
plt.ylabel('Accuracy')
plt.xlim((0,15))
plt.ylim((80,102))
plt.title('Comparison of Accuracy, C and Split Ration')
plt.legend(loc="lower right")
plt.show()
