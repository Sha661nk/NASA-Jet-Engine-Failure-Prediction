#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import math
import collections
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Tuning & Splitting Libraries
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Loading Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluation Liabraries
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, roc_auc_score, precision_score, recall_score, f1_score

import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

sns.set_style('darkgrid')

# Reading the data
jet_data = pd.read_csv('../data/train_FD001.txt', sep=" ", header=None)
jet_data.columns = ["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5"
                    ,"sensor6","sensor7","sensor8","sensor9","sensor10","sensor11","sensor12","sensor13"
                    ,"sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
                    ,"sensor20","sensor21","sensor22","sensor23"]

jet_data = jet_data[jet_data['id'] != 98]
jet_data = jet_data[jet_data['id'] != 99]
jet_data = jet_data[jet_data['id'] != 100]

jet_data.drop(['sensor22', 'sensor23'], axis=1, inplace=True)

jet_id_and_rul = jet_data.groupby(['id'])[["id" ,"cycle"]].max()
jet_id_and_rul.set_index('id', inplace=True)

def RUL_calculator(df, df_max_cycles):
    max_cycle = df_max_cycles["cycle"]
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='id', right_index=True)
    result_frame["RUL"] = result_frame["max_cycle"] - result_frame["cycle"] 
    result_frame.drop(['max_cycle'], axis=1, inplace=True)
    return result_frame

jet_data = RUL_calculator(jet_data, jet_id_and_rul)


jet_relevant_data = jet_data.drop(["cycle", "op1", "op2", "op3",
                                   "sensor1", "sensor5", "sensor6",
                                   "sensor10", "sensor16", "sensor18",
                                   "sensor19", "sensor14", "sensor13",
                                   "sensor12", "sensor11", "sensor9"], axis=1)

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(jet_relevant_data.drop(['id', 'RUL'], axis=1))
scaled_features = pd.DataFrame(scaled_features, columns=jet_relevant_data.drop(['id', 'RUL'], axis=1).columns)

scaled_features['id'] = jet_relevant_data['id']
scaled_features['RUL'] = jet_relevant_data['RUL']

cycle=30
data = scaled_features.copy()
data['label'] = data['RUL'].apply(lambda x: 1 if x <= cycle else 0)

y = data['label']
X = data.drop(['RUL', 'id', 'label'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

print('X_train shape : ', X_train.shape)
print('X_test shape : ', X_test.shape)
print('y_train shape : ', y_train.shape)
print('y_test shape : ', y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt 

def compare(y_pred, y_test):
    """
    Calculates and prints various classification metrics including confusion matrix, 
    classification report, and ROC AUC score, and plots the ROC curve. 
    """
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    # Classification Report
    cr = classification_report(y_test, y_pred)
    print("Classification Report:\n", cr)
    
    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC AUC Score:", roc_auc)
    
    # Plotting ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

# Example Usage
lr = LogisticRegression(C=1)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
compare(y_pred, y_test)

# Save the model
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(lr, file)

# Save the scaler
with open('minmax_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

