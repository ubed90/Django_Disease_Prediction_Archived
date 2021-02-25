#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle , joblib


# In[2]:


dataset_training = pd.read_csv('SHDPS_Training.csv')
dataset_testing = pd.read_csv('SHDPS_Testing.csv')
X_train = dataset_training.iloc[:, :-2].values
X_test = dataset_testing.iloc[:, :-1].values
Y_train = dataset_training.iloc[:, -2].values
Y_test = dataset_testing.iloc[:, -1].values


# In[3]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y_train_model = labelencoder.fit_transform(Y_train)
# Y_test = labelencoder.fit_transform(Y_test)


# In[4]:


np.set_printoptions(precision=2)
print(np.concatenate((Y_train.reshape(len(Y_train) , 1) , Y_train_model.reshape(len(Y_train_model) , 1)) , 1))


# In[5]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train , Y_train_model)


# In[6]:


Y_pred = classifier.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((Y_pred.reshape(len(Y_pred) , 1) , Y_test.reshape(len(Y_test) , 1)) , 1))


# In[7]:


accuracy_score = classifier.predict_proba(X_test)
accuracy_score = accuracy_score.max() * 100
accuracy_score


# In[8]:


symptoms = dataset_training.columns[ : -2]
print(len(symptoms))


# In[9]:


input_symtoms=[]
for x in range(0,len(symptoms)):
    input_symtoms.append(0)
for i in range(1 , len(symptoms) , 30):
    input_symtoms[i] = 1
input_symtoms.count(1)


# In[10]:


# NAME OF SYMPTOMS
for i in range(0 , len(input_symtoms)):
    if input_symtoms[i] == 1:
        print(symptoms[i])


# In[11]:


my_symptoms_pred = classifier.predict([input_symtoms])
my_symptoms_pred_acc_score = classifier.predict_proba([input_symtoms])


# In[12]:


for num , disease in zip(Y_train_model , Y_train):
    if int(num) == int(my_symptoms_pred[0]):
        print(disease)
        break
else:
    print('No Disease')
print(my_symptoms_pred_acc_score.max()*100)
