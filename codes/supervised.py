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
from sklearn.grid_search import GridSearchCV

# Parameters for SVM used for fine-tuning
param_grid = {
			'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
          }

# Read database
df = pd.read_csv("emotion.csv",header=0)
df = df.drop(["Person Id", "Person SubID"],axis=1)
df_train = df[df["Emotion"]!=-1] 					# Training Dataframe
df_test = df[df["Emotion"]==-1] 					# Testing Dataframe

# Final dataset
y = np.array(df_train["Emotion"])
X = np.array(df_train.drop(["Emotion"],axis=1))

# Split the dataset
# Split into training and validation data
X_train, X_cv, y_train, y_cv = train_test_split(X,y,test_size=0.2,random_state=15)

for i in range(1,17):
	# SVM, CNN, KNN, Random Forest, Naive Bayes, AdaBoost
	# from sklearn.svm import SVC
	# clf_svm = GridSearchCV(SVC(), param_grid)
	from sklearn.neighbors import KNeighborsClassifier
	clf_knn = KNeighborsClassifier(n_neighbors=i)
	# from sklearn.ensemble import RandomForestClassifier
	# clf_rf = RandomForestClassifier(n_estimators=10, min_samples_split=21)
	# from sklearn.naive_bayes import GaussianNB
	# clf_nb = GaussianNB()
	# from sklearn.neural_network import MLPClassifier
	# clf_nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 9), random_state=1)
	# from sklearn.ensemble import AdaBoostClassifier
	# clf_ada = AdaBoostClassifier(n_estimators=50)

	# print "Training SVM classifier..."
	# clf_svm.fit(X_train, y_train)
	# print clf_svm.best_params_
	# pred = clf_svm.predict(X_cv)
	# print accuracy_score(pred, y_cv)
	# print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
	print i, 
	clf_knn.fit(X_train, y_train)
	pred = clf_knn.predict(X_cv)
	# print clf_knn.best_params_
	print accuracy_score(pred, y_cv), 
	print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
	# print "Training Random Forest classifier..."
	# clf_rf.fit(X_train, y_train)
	# pred = clf_rf.predict(X_cv)
	# print accuracy_score(pred, y_cv)
	# print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
	# print "Training Gaussian Naive Bayes classifier..."
	# clf_nb.fit(X_train, y_train)
	# pred = clf_nb.predict(X_cv)
	# print accuracy_score(pred, y_cv)
	# print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
	# print "Training Multilayer Perceptron classifier..."
	# clf_nn.fit(X_train, y_train)
	# pred = clf_nn.predict(X_cv)
	# print accuracy_score(pred, y_cv)
	# print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
	# print "Training Adaboost classifier..."
	# clf_ada.fit(X_train, y_train)
	# pred = clf_ada.predict(X_cv)
	# print accuracy_score(pred, y_cv)
	# print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
