#!/usr/bin/env python
# coding: utf-8

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib , os

def read_dataset(dataset_copy):
    X_train = dataset_copy.iloc[ : , 1 : ].values
    Y = dataset_copy.iloc[ : , 0].values
    labelencoder = LabelEncoder()
    Y_train = labelencoder.fit_transform(Y)
    return X_train , Y_train , Y


def train_SVC_model(dataset_copy):
    X_train , Y_train , Y = read_dataset(dataset_copy)
    classifier = SVC(kernel = 'rbf' , probability = True , random_state = 0)
    classifier.fit(X_train , Y_train)
    joblib.dump(classifier , os.getcwd() + '\Kernel_SVM\Kernel_SVM_Model.sav')

