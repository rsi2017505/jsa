#!/usr/bin/python

import joblib
from sklearn.metrics import confusion_matrix

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Import pandas
import pandas as pd
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# and later you can load it
clf = joblib.load('rf.pkl')

# making prediction for out of sample data 
newdata = pd.read_csv("/path/to/the/input/file/test.csv") 
preds = clf.predict(newdata) 
print("Predictions of optimization Level:", preds)

actual=['O2', 'O2', 'Ofast', 'O3', 'O2', 'O1', 'O2', 'O3']

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(actual, preds))

print(confusion_matrix(actual,preds))
