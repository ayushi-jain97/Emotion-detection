'''
Hyper-Parameter Tuning for Random Forest Classifier

# Libraries Used
    pandas,sklearn,numpy

# Variables Used

'''

# Libraries used
import pandas as pd                                                             # To read database
from sklearn.cross_validation import train_test_split                           # To split database
from sklearn.metrics import accuracy_score, precision_recall_fscore_support     # Result of model on database
import numpy as np                                                              # Mathematical analysis
import matplotlib.pyplot as plt                                                 # Plotting result
from mpl_toolkits.mplot3d import Axes3D                                         # 3D Axes

# Read database
df = pd.read_csv("emotion.csv",header=0)
df = df.drop(["Person Id", "Person SubID"],axis=1)
df_train = df[df["Emotion"]!=-1]                    # Training Dataframe
df_test = df[df["Emotion"]==-1]                     # Testing Dataframe

# Final dataset
y = np.array(df_train["Emotion"])
x = np.array(df_train.drop(["Emotion"],axis=1))

# Split the dataset
# Split into training and validation data
X_train, X_cv, y_train, y_cv = train_test_split(x,y,test_size=0.2,random_state=42)

X,Y,Z = list(),list(),list()
from sklearn.ensemble import RandomForestClassifier
for i in range(5,105,5):
    for j in range(1,7):
        clf_rf = RandomForestClassifier(n_estimators=i, min_samples_split=2**j)
        print "Training Random Forest classifier..."
        clf_rf.fit(X_train, y_train)
        pred = clf_rf.predict(X_cv)
        Z=accuracy_score(pred, y_cv)
        print precision_recall_fscore_support(pred, y_cv, average='weighted', labels=list(range(8)))
        X.append(i)
        Y.append(2**j)
        Z.append(Z*100)

#Plot the 3D scatter plot for Random Forest Classifier using Axes3D module.
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X,Y,Z,zdir='z')

#Provide attributes for the plot
ax.set_xlim(0, 100)
ax.set_ylim(0, 70)
ax.set_zlim(70,100)
ax.set_xlabel('The no. of trees in the forest')
ax.set_ylabel('Minimum No. of samples required to split an internal node')
ax.set_zlabel('Accuracy obtained.')

#Display the plot.
plt.show()