'''
CNN Deep Learning approach on CK+ Dataset, using Keras library with tensorflow backend

# Libraries used
    pandas, sklearn, numpy, keras

# Variables
    batch_size : size of batch
    num_classes : number of classes in classification
    epochs : number of epochs run
    df : pandas dataframe
    df_labelled : Dataframe of labelled data
    df_unlabelled : Dataframe of unlabelled data
    X : inputs
    y : labels
    model : Sequential Keras model
'''
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D
import keras


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y-1] = 1
    return categorical


# Dataset info
batch_size = 32
num_classes = 7
epochs = 200

# Read database
df = pd.read_csv("emotion.csv",header=0)
df = df.drop(["Person Id", "Person SubID"],axis=1)
df_train = df[df["Emotion"]!=-1]                    # Training Dataframe
df_test = df[df["Emotion"]==-1]                     # Testing Dataframe

# Final Dataset
y = np.array(df_train["Emotion"])
X = np.array(df_train.drop(["Emotion"],axis=1))
X = np.expand_dims(X, axis=2)

# Split the dataset
# Split into training and validation data
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Using sequential Keras model
model = Sequential()
model.add(Convolution1D(4, 3, input_shape=(64,1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)
