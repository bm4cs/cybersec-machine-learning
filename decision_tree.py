# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn . tree import DecisionTreeClassifier

df = pd. read_csv ('network_data_numbers.csv')

print (df. head (3))

y = df['label']. values
print ('Y Shape', y. shape)

X = df. drop ('label', axis =1) . values
print ('X Shape', X. shape)

# divide data into training and testing sets
X_train , X_test , y_train , y_test = train_test_split (X, y, test_size =0.2, random_state =1)

# select model
model = DecisionTreeClassifier ()

# train the model
model.fit ( X_train , y_train )

# prediction using the testing phase
y_pred = model.predict ( X_test )

# Measuring performance using Accuracy 
print ("Accuracy", accuracy_score (y_pred , y_test ))

#Measuring the performance using Confusion Matrix
print ("Confusion Matrix", confusion_matrix (y_pred , y_test ))