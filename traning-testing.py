#!/usr/bin/python

import numpy as np
from sklearn.metrics import confusion_matrix
import joblib

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Import pandas
import pandas as pd

# Import train_test_split function
from sklearn.model_selection import train_test_split

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=50)

#Load dataset
mydata = pd.read_csv("/home/systemlab/ml-python/random-forest/temp.csv")

#provided your csv has header row, and the label column is named "Label"
target = mydata["Label"] 

print("\nFeatures:", mydata.columns[:-1])
print("\nLabels: O1; O2; O3; Ofast;")
print("Shape of data :",mydata.shape)

#print(mydata[0:5])

X = mydata[mydata.columns[:-1]] 
y = mydata[mydata.columns[-1]] 

# printing first 5 rows of feature matrix 
#print("\nFeature matrix:\n", X.head()) 
  
# printing first 5 values of response vector 
#print("\nResponse vector:\n", y.head())

 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
  
# printing the shapes of the new X objects 
print("Features Training data size:",X_train.shape) 
print("Feature Testing data size:",X_test.shape) 
  
# printing the shapes of the new y objects 
print("Label Training data size:",y_train.shape) 
print("Label Testing data size:",y_test.shape)

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#print(y_pred)
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# making prediction for out of sample data 
#newdata = pd.read_csv("/home/systemlab/ml_python/sample.csv") 
#preds = clf.predict(newdata) 
#print("Predictions of optimization Level:", preds) 

print(confusion_matrix(y_test, y_pred))

# now you can save it to a file
joblib.dump(clf, 'rf.pkl') 

#feature_imp = pd.Series(clf.feature_importances_,index=mydata.columns[:-1]).sort_values(ascending=False)
#print(feature_imp)


