# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:59:22 2020

@author: irzar
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# read dataset with its labels
names = ['degree','caprice','topic','lmt','lpss','pb']
dataset = pd.read_csv("kohkiloyeh.csv", names=names)

# convert category from string to numeric
degrees = dataset['degree']
caprices= dataset['caprice']
topics = dataset['topic']
lmts = dataset['lmt']
lpsss = dataset['lpss']
pbs = dataset['pb']

le = preprocessing.LabelEncoder()
dg_en = le.fit_transform(degrees)
#high=0;low=1;med=2
cp_en = le.fit_transform(caprices)
#left=0;middle=1;rigth=2
tp_en = le.fit_transform(topics)
#impression=0;news=1;political=2;scientific=3;tourism=4
lmt_en = le.fit_transform(lmts)
#no=0;yes=1
lpss_en = le.fit_transform(lpsss)
#no=0;yes=1
pb_en = le.fit_transform(pbs)
#no=0;yes=1

# Seperate features and class (category)
fitur_gabung = np.array([dg_en,cp_en,tp_en,lmt_en,lpss_en])
X_data = np.ndarray.transpose(fitur_gabung)
y_data = pb_en

# divide data into data training&testing
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)

model = KNeighborsClassifier(n_neighbors=7, weights="distance")
model.fit(X_train, y_train)

# getting prediction from data testing
predicted = model.predict(X_test)

# print prediction
print('- Classification using KNN -')
print("Hasil Klasifikasi (KNN) dengan Data Testing          : \n", predicted)
print()

# print category(class) from real data
print("Hasil Klasifikasi yang benar dengan Data Training    : \n", y_test)
print()

#print presentation of prediciton error
error = ((y_test != predicted).sum()/len(predicted))*100
print("Error Prediction = %.2f" %error,"%")

#print presentation of accuracy
accuracy = 100-error
print("Accuracy         = %.2f" %accuracy,"%")
print()

# model evaluation
def Conf_matrix(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(y_pred)):
        if y_actual[i]==y_pred[i]==1:
            TP += 1
        if y_pred[i]==1 and y_actual[i] !=y_pred[i]:
            FP += 1
        if y_actual[i]==y_pred[i]==0:
            TN += 1
        if y_pred[i]==0 and y_actual[i]!= y_pred[i]:
            FN += 1           
    return (TP, FN, TN, FP)

#hold out estimation evaluation
TP, FN, TN, FP = Conf_matrix(y_test, predicted)

print('- Model Evaluation Hold Out Estimation -')
print('Accuracy     = ', (TP+TN)/(TP+TN+FP+FN))
print('Sensitivity  = ', TP/(TP+FN))
print('Specificity  = ', TN/(TN+FP))