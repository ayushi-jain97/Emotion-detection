import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import sklearn.metrics
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle


df = pd.read_csv("emotion.csv",header=0)

df = df.drop(["Person Id", "Person SubID"],axis=1)
df_train = df[df["Emotion"]!=-1]
df_test = df[df["Emotion"]==-1]

y = np.array(df_train["Emotion"])
X = np.array(df_train.drop(["Emotion"],axis=1))
#
y = label_binarize(y, classes=[1, 2,3,4,5,6,7])
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

X_train, X_cv, y_train, y_cv = train_test_split(X,y,test_size=0.2,random_state=42)

print X_train.shape, X_cv.shape

# SVM, CNN, ANN, KNN, Random Forest, Naive Bayes
from sklearn.svm import SVC
clf_svm =  OneVsRestClassifier(SVC(kernel="rbf", C=10000))
from sklearn.neighbors import KNeighborsClassifier
clf_knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3))
from sklearn.ensemble import RandomForestClassifier
clf_rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=10, min_samples_split=50))
from sklearn.naive_bayes import GaussianNB
clf_nb = OneVsRestClassifier(GaussianNB())
from sklearn.neural_network import MLPClassifier
clf_nn = OneVsRestClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 4), random_state=1))
from sklearn.ensemble import AdaBoostClassifier
clf_ada = OneVsRestClassifier(AdaBoostClassifier(n_estimators=50))

n_classes=7

print "Training SVM classifier.."
clf_svm.fit(X_train, y_train)
pred = clf_svm.decision_function(X_cv)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_cv[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
	

##plt.figure()
##lw = 2
##plt.plot(fpr[2], tpr[2], color='darkorange',
##         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
##plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
##plt.xlim([0.0, 1.0])
##plt.ylim([0.0, 1.05])
##plt.xlabel('False Positive Rate')
##plt.ylabel('True Positive Rate')
##plt.title('Receiver operating characteristic example')
##plt.legend(loc="lower right")
##plt.show()
##
##
##
###print "Training Adaboost classifier..."
#clf_ada.fit(X_train, y_train)
#pred = clf_ada.predict(X_cv)
v={0:'anger',1:'contempt',2:'disgust',3:'fear',4:'happy',5:'sadness',6:'surprise'}
fpr["micro"], tpr["micro"], _ = roc_curve(y_cv.ravel(), pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
lw=2
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','green','purple','lightgreen'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(v[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
