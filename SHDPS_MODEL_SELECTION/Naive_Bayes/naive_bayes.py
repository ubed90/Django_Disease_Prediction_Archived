#!/usr/bin/env python
# coding: utf-8

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB , MultinomialNB
import joblib , os


def train_naive_bayes_model(dataset_copy):
    X_train , Y_train , Y = read_dataset(dataset_copy)
    classifier_1 = GaussianNB()
    classifier_2 = MultinomialNB()
    classifier_1.fit(X_train , Y_train)
    classifier_2.fit(X_train , Y_train)
    joblib.dump(classifier_1 , os.getcwd() + '\\Naive_Bayes\\Gaussian_Bayes_Model.sav')
    joblib.dump(classifier_2 , os.getcwd() + '\\Naive_Bayes\\Multinomial_Bayes_Model.sav')

def read_dataset(dataset_copy):
    X_train = dataset_copy.iloc[ : , 1 : ].values
    Y = dataset_copy.iloc[ : , 0].values
    labelencoder = LabelEncoder()
    Y_train = labelencoder.fit_transform(Y)
    return X_train , Y_train , Y
