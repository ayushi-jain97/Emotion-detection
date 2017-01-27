import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
df = pd.read_csv("emotion.csv",header=0)

df = df.drop(["Person Id", "Person SubID"],axis=1)
df_train = df[df["Emotion"]!=-1]
df_test = df[df["Emotion"]==-1]

y = np.array(df_train["Emotion"])
X = np.array(df_train.drop(["Emotion"],axis=1))

X_train, X_cv, y_train, y_cv = train_test_split(X,y,test_size=0.2,random_state=42)

print X_train.shape, X_cv.shape

# SVM, CNN, ANN, KNN, Random Forest, Naive Bayes
from sklearn.svm import SVC
clf_svm = SVC(kernel="rbf", C=10000)
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=10, min_samples_split=50)
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
from sklearn.neural_network import MLPClassifier
clf_nn = MLPClassifier(solver='lbgfs', alpha=1e-5, hidden_layer_sizes=(5, 4), random_state=1)
from sklearn.ensemble import AdaBoostClassifier
clf_ada = AdaBoostClassifier(n_estimators=50)

print "Training SVM classifier..."
clf_svm.fit(X_train, y_train)
pred = clf_svm.predict(X_cv)
print accuracy_score(pred, y_cv)
print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
print "Training K Nearest Neighbours classifier..."
clf_knn.fit(X_train, y_train)
pred = clf_knn.predict(X_cv)
print accuracy_score(pred, y_cv)
print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
print "Training Random Forest classifier..."
clf_rf.fit(X_train, y_train)
pred = clf_rf.predict(X_cv)
print accuracy_score(pred, y_cv)
print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
print "Training Gaussian Naive Bayes classifier..."
clf_nb.fit(X_train, y_train)
pred = clf_nb.predict(X_cv)
print accuracy_score(pred, y_cv)
print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
print "Training Multilayer Perceptron classifier..."
clf_nn.fit(X_train, y_train)
pred = clf_nn.predict(X_cv)
print accuracy_score(pred, y_cv)
print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
print "Training Adaboost classifier..."
clf_ada.fit(X_train, y_train)
pred = clf_ada.predict(X_cv)
print accuracy_score(pred, y_cv)
print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
