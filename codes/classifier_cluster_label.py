'''
Cluster-Based Semi-Supervised Learning Approach on CK+ dataset

# Libraries Used
	pandas, sklearn, numpy

# Variables
	clf_svm : SVM classifier
	df : pandas dataframe
	df_labelled : Dataframe of labelled data
	df_unlabelled : Dataframe of unlabelled data
	km : K-means model
	X : inputs
	y : labels
	clusters : Cluster formed by K-means
	pred : predictions
'''

# Libraries used
import pandas as pd 															# To read database
from sklearn.cross_validation import train_test_split 							# To split database
from sklearn.metrics import accuracy_score, precision_recall_fscore_support 	# Result of model on database
import numpy as np 																# Mathematical analysis
from sklearn.svm import SVC 													# To apply SVM on data
from sklearn.cluster import KMeans 												# To apply KMeans

param_grid = {
         'C': [1e-2, 1e-1, 1e0, 1e2, 1e1, 1e3, 5e3, 1e4, 5e4,1e5,450000],
         'kernel': ['linear', 'rbf']
          }

def supervised(train_x,train_y,test_x):
	'''
	Given the dataset of the cluster containing both labelled and unlabelled data
	
	# Arguements
		train_x : input of labelled data
		train_y : output of labelled data
		test_x : input of unlabelled data

	# Returns
		pred_y : predicted output corresponding to unlabelled data
	'''
	pred_y=[]
	if(len(set(train_y))==1):
		pred_y=[train_y[0] for i in range(len(test_x))]
	else:
		try:
			clf_svm.fit(train_x, train_y)
			pred_y = clf_svm.predict(test_x)
		except Exception:
			pass
	return pred_y


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

X = np.array(df.drop(["Emotion"],axis=1))
y = np.array(df["Emotion"])

comp_data_x=[]
comp_data_y=[]

# Applyying Cluter-Based approach to get complete dataset
km = KMeans(n_clusters=10).fit(X)
pred = km.predict(X)

clusters={unique_pred:[] for unique_pred in np.unique(np.array(pred))}

for i in range(len(pred)):
	clusters[pred[i]].append([X[i],y[i]])

for i in clusters.keys():
	train_cluster=[]
	test_cluster=[]
	train_label=[]
	for j in x[i]:
		if(j[1]==-1):
			test_cluster.append(j[0])
		else:
			train_cluster.append(j[0])
			train_label.append(j[1])
	test_label = supervised(train_cluster,train_label,test_cluster)
	for j in range(len(test_cluster)):
		comp_data_x.append(test_cluster[j])
		comp_data_y.append(test_label[j])
	for j in range(len(train_cluster)):
		comp_data_x.append(train_cluster[j])
		comp_data_y.append(train_label[j])

# Final dataset
X=np.array(comp_data_x)
y=np.array(comp_data_y)

# Split the dataset
# Split into training and validation data
X_train, X_cv, y_train, y_cv = train_test_split(X,y,test_size=0.2,random_state=42)

# Train the dataset, and get results for the dataset
clf_svm.fit(X_train, y_train)
pred = clf_svm.predict(X_cv)
print accuracy_score(pred, y_cv)
print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
