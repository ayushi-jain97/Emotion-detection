# Emotion-detection
Identify the emotion (neutral, anger, contempt, disgust, fear, happy, sadness, surprise) in a given static image. This is an example of a Supervised Machine Learning problem.

# Prerequisites
* CK+ AU Coded Database is available for free. It can be downloaded from here. http://www.consortium.ri.cmu.edu/ckagree/
* Install the following python libraries:
    * pandas
    * sklearn
    * numpy

# Getting Started
Run read_database.py
```python
python read_database.py
```
This will create a file emos.csv that can be opened and edited in Excel. This file has all features (AU Codes) and labels         (Emotions). 

Next run classifier_test.py
```python
python classifier_test.py
```
See the ouput and compare accuracy, recall, precision, F-measure of different classifiers.
Additionally, Confusion Matrix can be printed
```python
from pandas_ml import ConfusionMatrix as cm
print cm(y_true,y_predicted)
```


