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
import matplotlib.pyplot as plt                                                 # Plotting result

# Read database
df = pd.read_csv("emotion.csv",header=0)
df = df.drop(["Person Id", "Person SubID"],axis=1)
df_train = df[df["Emotion"]!=-1]                    # Training Dataframe
df_test = df[df["Emotion"]==-1]                     # Testing Dataframe

# Final dataset
y = np.array(df_train["Emotion"])
X = np.array(df_train.drop(["Emotion"],axis=1))

# Parameter tuning
Cvals =[0.01,0.05,0.1,0.5,1,2,4,8,10,15]
R=[]
for i in range(len(Cvals)):
    for j in range(2,5):
        for k in range(1,3):
            X_train, X_cv, y_train, y_cv = train_test_split(X,y,test_size=(j/10.0),random_state=42)
            print X_train.shape, X_cv.shape
            if(k==1):
                clf_svm = SVC(kernel="linear",C=Cvals[i])
            else:
                clf_svm = SVC(kernel="rbf",C=Cvals[i])
            print "Training SVM classifier..."
            clf_svm.fit(X_train, y_train)
            pred = clf_svm.predict(X_cv)
            Z = accuracy_score(pred, y_cv)
            r=j/10.0
            tup = (Cvals[i],r,Z,k)
            R.append(str(tup))         

X=[[[],[],[],[]],[[],[],[],[]]]
accuracy=[[[],[],[],[]],[[],[],[],[]]]
for i in range(0,len(R)):
    K=R[i][1:len(R[i])-1]
    labels =K.strip().split(',')
    Y=int(float(labels[1])*10) - 1
    X[int(labels[3]) -1][Y].append(float(labels[0]))
    accuracy[int(labels[3]) -1][Y].append(100*float(labels[2]))

# Plotting data
colors=["navy","darkorange","red","green","yellow","magenta"]
plt.figure()
k=0
for j in range(2):
    for i in range(1,4):
        if(j==0):
            s="Split-ratio "+ str((i+1)/10.0) + " for linear kernel "
        else:
            s="Split-ratio "+ str((i+1)/10.0) + " for RBF kernel "
        plt.plot(X[j][i],accuracy[j][i],colors[k],label=s)
        k= k+1
    
plt.xlabel('Value of C')
plt.ylabel('Accuracy')
plt.ylim((80,102))
plt.title('Comparison of Accuracy, C and Split Ratio for Different Kernels')
plt.legend(loc="lower right")
plt.show()
