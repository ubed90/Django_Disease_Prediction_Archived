#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import joblib


# In[16]:


dataset_training = pd.read_csv('SHDPS_Training.csv')
dataset_testing = pd.read_csv('SHDPS_Testing.csv')
X_train = dataset_training.iloc[:, :-2].values
X_test = dataset_testing.iloc[:, :-1].values
Y_train = dataset_training.iloc[:, -2].values
Y_test = dataset_testing.iloc[:, -1].values


# In[17]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y_train_model = labelencoder.fit_transform(Y_train)
# Y_test = labelencoder.fit_transform(Y_test)


# In[18]:


np.set_printoptions(precision=2)
print(np.concatenate((Y_train.reshape(len(Y_train) , 1) , Y_train_model.reshape(len(Y_train_model) , 1)) , 1))


# In[19]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)
classifier.fit(X_train , Y_train_model)
# classifier = joblib.load('Decision_Tree_Model.sav')


# In[20]:


Y_pred = classifier.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((Y_pred.reshape(len(Y_pred) , 1) , Y_test.reshape(len(Y_test) , 1)) , 1))


# In[21]:


accuracy_score = classifier.predict_proba(X_test)
accuracy_score = accuracy_score.max() * 100
accuracy_score


# In[22]:


symptoms = dataset_training.columns[ : -2]
print(len(symptoms))


# In[23]:


input_symtoms=[]
for x in range(0,len(symptoms)):
    input_symtoms.append(0)
for i in range(1 , len(symptoms) , 30):
    input_symtoms[i] = 1
input_symtoms.count(1)


# In[24]:


# NAME OF SYMPTOMS
for i in range(0 , len(input_symtoms)):
    if input_symtoms[i] == 1:
        print(symptoms[i])


# In[25]:


my_symptoms_pred = classifier.predict([input_symtoms])
my_symptoms_pred_acc_score = classifier.predict_proba([input_symtoms])


# In[26]:


for num , disease in zip(Y_train_model , Y_train):
    if int(num) == int(my_symptoms_pred[0]):
        print(disease)
        break
else:
    print('No Disease')
print(my_symptoms_pred_acc_score.max()*100)


# In[27]:


# joblib.dump(classifier , 'Decision_Tree_Model.sav')

