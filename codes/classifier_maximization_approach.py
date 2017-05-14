'''
Expectation Maximization Semi-Supervised Learning Approach on CK+ dataset

# Libraries Used
    pandas, sklearn, numpy

# Variables
    clf_svm : SVM classifier
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
from sklearn.svm import SVC                                                     # To apply SVM

param_grid = {
         'C': [1e-2, 1e-1, 1e0, 1e2, 1e1, 1e3, 5e3, 1e4, 5e4,1e5,450000],
         'kernel': ['linear', 'rbf']
          }

# SVM classifier with linear kernel and C = 0.1
clf_svm = SVC(kernel="linear", C=0.1)

# Read database
df = pd.read_csv("emotion.csv",header=0)
df = df.drop(["Person Id", "Person SubID"],axis=1)
df_labelled = df[df["Emotion"]!=-1]
df_unlabelled = df[df["Emotion"]==-1]

# Seperate labelled and unlabelled data
y_labelled = df_labelled["Emotion"]
X_labelled = df_labelled.drop(["Emotion"],axis=1)

y_unlabelled = df_unlabelled["Emotion"]
X_unlabelled = df_unlabelled.drop(["Emotion"],axis=1)

# Apply Expectation Maximization
clf_svm.fit(X_labelled,y_labelled)
pred = clf_svm.predict(X_unlabelled)
df["Emotion"][df["Emotion"]==-1]=pred

# Final dataset
X=df.drop(["Emotion"],axis=1)
y=df["Emotion"]

# Split the dataset
# Split into training and validation data
X_train, X_cv, y_train, y_cv = train_test_split(X,y,test_size=0.2,random_state=42)

# SVM
clf_svm.fit(X_train, y_train)
pred = clf_svm.predict(X_cv)
print accuracy_score(pred, y_cv)
print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
