#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
import joblib


# In[28]:


dataset_training = pd.read_csv('SHDPS_Training.csv')
dataset_testing = pd.read_csv('SHDPS_Testing.csv')
X_train = dataset_training.iloc[:, :-2].values
X_test = dataset_testing.iloc[:, :-1].values
Y_train = dataset_training.iloc[:, -2].values
Y_test = dataset_testing.iloc[:, -1].values


# In[29]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y_train_model = labelencoder.fit_transform(Y_train)
# Y_test = labelencoder.fit_transform(Y_test)


# In[30]:


np.set_printoptions(precision=2)
print(np.concatenate((Y_train.reshape(len(Y_train) , 1) , Y_train_model.reshape(len(Y_train_model) , 1)) , 1))


# In[31]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10 , criterion = 'entropy' , random_state = 0)
classifier.fit(X_train , Y_train_model)
# classifier = joblib.load('Random_Forest_Model.sav')


# In[32]:


Y_pred = classifier.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((Y_pred.reshape(len(Y_pred) , 1) , Y_test.reshape(len(Y_test) , 1)) , 1))


# In[33]:


accuracy_score = classifier.predict_proba(X_test)
accuracy_score = accuracy_score.max() * 100
accuracy_score


# In[34]:


symptoms = dataset_training.columns[ : -2]
print(len(symptoms))


# In[35]:


input_symtoms=[]
for x in range(0,len(symptoms)):
    input_symtoms.append(0)
# for i in range(1 , len(symptoms) , 30):
#     input_symtoms[i] = 1
faiyaz_symptoms = ['joint_pain' , 'vomiting' , 'fatigue' , 'yellowish_skin' , 'dark_urine' , 'loss_of_appetite' , 'abdominal_pain' , 'yellowing_of_eyes']
for index , symptom in enumerate(symptoms):
    for faiyaz_symptom in faiyaz_symptoms:
        if symptom == faiyaz_symptom:
            input_symtoms[index] = 1
input_symtoms.count(1)


# In[36]:


# NAME OF SYMPTOMS
for i in range(0 , len(input_symtoms)):
    if input_symtoms[i] == 1:
        print(symptoms[i])


# In[37]:


my_symptoms_pred = classifier.predict([input_symtoms])
my_symptoms_pred_acc_score = classifier.predict_proba([input_symtoms])


# In[38]:


for num , disease in zip(Y_train_model , Y_train):
    if int(num) == int(my_symptoms_pred[0]):
        print(disease)
        break
else:
    print('No Disease')
print(my_symptoms_pred_acc_score.max()*100)


# In[39]:


# joblib.dump(classifier , 'Random_Forest_Model.sav')

