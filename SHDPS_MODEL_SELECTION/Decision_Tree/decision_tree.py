#!/usr/bin/env python
# coding: utf-8

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib , os

def read_dataset(dataset_copy):
    X_train = dataset_copy.iloc[ : , 1 : ].values
    Y = dataset_copy.iloc[ : , 0].values
    labelencoder = LabelEncoder()
    Y_train = labelencoder.fit_transform(Y)
    return X_train , Y_train , Y


def train_decision_tree_model(dataset_copy):
    X_train , Y_train , Y = read_dataset(dataset_copy)
    classifier = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)
    classifier.fit(X_train , Y_train)
    joblib.dump(classifier , os.getcwd() + '\Decision_Tree\Decision_Tree_Model.sav')
