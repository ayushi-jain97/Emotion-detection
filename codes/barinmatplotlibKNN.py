'''
Plotting a bar plot for K-Nearest Neighbour Classifier

Libraries Used
matplotlib,numpy

Variables Used
X= values of K for KNN classifier
Y=Accuracy values for KNN classifier
Z=Precision Values for KNN classifier
K=Recall Values for KNN classifier

'''
import matplotlib.pyplot as plt
import numpy as np

#Create a plot using matplotlib
multiple_bars = plt.figure()

#Read the file containing the data for plotting
file2 = open('KNNTuning.txt','r')
X=file2.read()
file2.close()
R=X.strip().split('\n')

#Obtain Accuracy,Precision, Recall Values for values of K
X,Y,Z,K = list(),list(),list(),list()
for i in range(0,len(R)):
    labels =R[i].strip().split(',')
    if(i%2 == 0):
        X.append(float(labels[0]))
        Y.append(100.0*float(labels[1]))
        Z.append(100.0*float(labels[2]))
        K.append(100.0*float(labels[3]))
    
X=np.array(X)

#Plot the accuracy,precision and recall values for values of X.
ax = plt.subplot(111)
rect1=ax.bar(X-0.4,Y,width=0.4,color="#E91E63",align='center')
rect2=ax.bar(X, Z,width=0.4,color="#64B5F6",align='center')
rect3 =ax.bar(X+0.4,K,width=0.4,color="#E0E0E0",align='center')

#Set the attributes for the plot
plt.ylim((min(min(Y),min(min(Z),min(K)))-1,max(max(Y),max(max(Z),max(K)))+1))
ax.set_ylabel('Percentage')
ax.set_title('Accuracy, Precision and Recall Values')
ax.set_xticks((X))
ax.set_xlabel('Value of K for K-Nearest Neighbours')
ax.legend((rect1[0], rect2[0],rect3[0]), ('Accuracy', 'Precision','Recall'))

#Display the plot
plt.show()

