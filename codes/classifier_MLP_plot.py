'''
Plotting a bar plot for K-Nearest Neighbour Classifier

# Libraries Used
	pandas, matplotlib, sklearn, numpy

# Variables Used
	df : pandas dataframe
	df_labelled : Dataframe of labelled data
	df_unlabelled : Dataframe of unlabelled data
	X : inputs
	Y : labels
	pred : predictions
	x : Number of hidden layers
	y : Number of units in hidden layers
	z : Accuracy
'''

# Libraries used
import pandas as pd                                                             # To read database
from sklearn.cross_validation import train_test_split                           # To split database
from sklearn.metrics import accuracy_score, precision_recall_fscore_support     # Result of model on database
import numpy as np                                                              # Mathematical analysis
import matplotlib.pyplot as plt 												# Plotting result
from mpl_toolkits.mplot3d import Axes3D 										# 3D Axes

# Read database
df = pd.read_csv("emotion.csv",header=0)
df = df.drop(["Person Id", "Person SubID"],axis=1)
df_train = df[df["Emotion"]!=-1] 					# Training Dataframe
df_test = df[df["Emotion"]==-1] 					# Testing Dataframe

# Final dataset
Y = np.array(df_train["Emotion"])
X = np.array(df_train.drop(["Emotion"],axis=1))

# Split the dataset
# Split into training and validation data
X_train, X_cv, y_train, y_cv = train_test_split(X,Y,test_size=0.2,random_state=42)

x, y, z = [], [], []
for i in range(1,11):
	for j in range(1,11):
		print i,j
		from sklearn.neural_network import MLPClassifier
		clf_nn = MLPClassifier(solver='lbgfs', alpha=1e-5, hidden_layer_sizes=(i, j), random_state=1)
		print "Training Multilayer Perceptron classifier..."
		clf_nn.fit(X_train, y_train)
		pred = clf_nn.predict(X_cv)
		x.append(i)
		y.append(j)
		z.append(accuracy_score(pred, y_cv)*100)

# Plot the results
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x,y,z,zdir='z')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0,100)
ax.set_xlabel('Number of hidden layers')
ax.set_ylabel('Number of elements in each hidden layer')
ax.set_zlabel('Accuracy obtained.')

plt.show()
