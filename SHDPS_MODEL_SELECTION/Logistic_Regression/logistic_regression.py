#!/usr/bin/env python
# coding: utf-8

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib , os


def train_Logistic_regression_model(dataset_copy):
    X_train , Y_train , Y = read_dataset(dataset_copy)
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train , Y_train)
    joblib.dump(classifier , os.getcwd() + '\Logistic_Regression\Logistic_Regression_Model.sav')


def read_dataset(dataset_copy):
    X_train = dataset_copy.iloc[ : , 1 : ].values
    Y = dataset_copy.iloc[ : , 0].values
    labelencoder = LabelEncoder()
    Y_train = labelencoder.fit_transform(Y)
    return X_train , Y_train , Y

