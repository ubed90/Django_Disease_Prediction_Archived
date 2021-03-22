#!/usr/bin/env python
# coding: utf-8

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import joblib , os


def read_dataset(dataset_copy):
    X_train = dataset_copy.iloc[ : , 1 : ].values
    Y = dataset_copy.iloc[ : , 0].values
    labelencoder = LabelEncoder()
    Y_train = labelencoder.fit_transform(Y)
    return X_train , Y_train , Y


def train_KNN_model(dataset_copy):
    X_train , Y_train , Y = read_dataset(dataset_copy)
    classifier = KNeighborsClassifier(n_neighbors = 30 , metric = 'minkowski' , p = 2)
    classifier.fit(X_train , Y_train)
    joblib.dump(classifier , os.getcwd() + '\K_Nearest_Neighbors\KNN_Model.sav')

