'''
Co-Training Semi-Supervised Learning Approach on CK+ dataset

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
    k : Parameter for co-training
'''

# Libraries used
import pandas as pd                                                             # To read database
from sklearn.cross_validation import train_test_split                           # To split database
from sklearn.metrics import accuracy_score, precision_recall_fscore_support     # Result of model on database
import numpy as np                                                              # Mathematical analysis
from sklearn.svm import SVC                                                     # To apply SVM
from sklearn.grid_search import GridSearchCV                                    # For Hyperparameter tuning

param_grid = {
         'C': [1e-2, 1e-1, 1e0, 1e2, 1e1, 1e3, 5e3, 1e4, 5e4,1e5,450000],
         'kernel': ['linear', 'rbf']
          }

clf_svm1 = GridSearchCV(SVC(), param_grid) # Classifier 1
clf_svm2 = GridSearchCV(SVC(), param_grid) # Classifier 2

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

#COTRAINING
k=4
X_train1, X_train2, y_train1, y_train2 = train_test_split(np.array(X_labelled),np.array(y_labelled),test_size=0.5,random_state=42)
length1=X_unlabelled.shape[0]
X_u=np.array(X_unlabelled)
X_unlabelled1=np.array(X_u[0:(length1/2)])
X_unlabelled2=np.array(X_u[(length1/2):])
np.random.shuffle(X_unlabelled1)
np.random.shuffle(X_unlabelled2)

low,high=0,k
while(low<length1/2):
    clf_svm1.fit(X_train1,y_train1)
    clf_svm2.fit(X_train2,y_train2)
    X1=X_unlabelled1[low:high]
    X2=X_unlabelled2[low:high]
    pred1=clf_svm1.predict(X1)
    pred2=clf_svm2.predict(X2)
    
    X_train1=np.concatenate((X_train1,X2),axis=0)
    X_train2=np.concatenate((X_train2,X1),axis=0)
    y_train1=np.concatenate((y_train1,pred2),axis=0)
    y_train2=np.concatenate((y_train2,pred1),axis=0)
    low,high=low+k,high+k
    
    
# Final dataset
X=np.concatenate((X_train1,X_train2),axis=0)
y=np.concatenate((y_train1,y_train2),axis=0)

# Split the dataset
X_train, X_cv, y_train, y_cv = train_test_split(X,y,test_size=0.2,random_state=42)

# SVM
clf_svm = GridSearchCV(SVC(), param_grid)
clf_svm.fit(X_train, y_train)
print clf_svm.best_params_
pred = clf_svm.predict(X_cv)
print accuracy_score(pred, y_cv)
print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
