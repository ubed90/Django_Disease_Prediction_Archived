#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
import joblib


# In[53]:


dataset_training = pd.read_csv('SHDPS_Training.csv')
dataset_testing = pd.read_csv('SHDPS_Testing.csv')
X_train = dataset_training.iloc[:, :-2].values
X_test = dataset_testing.iloc[:, :-1].values
Y_train = dataset_training.iloc[:, -2].values
Y_test = dataset_testing.iloc[:, -1].values


# In[54]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y_train_model = labelencoder.fit_transform(Y_train)
# Y_test = labelencoder.fit_transform(Y_test)


# In[55]:


np.set_printoptions(precision=2)
print(np.concatenate((Y_train.reshape(len(Y_train) , 1) , Y_train_model.reshape(len(Y_train_model) , 1)) , 1))


# In[56]:


from sklearn.naive_bayes import GaussianNB , MultinomialNB
classifier_1 = GaussianNB()
classifier_2 = MultinomialNB()
classifier_1.fit(X_train , Y_train_model)
classifier_2.fit(X_train , Y_train_model)
# classifier_1 = joblib.load('Gaussian_Bayes_Model.sav')
# classifier_2 = joblib.load('Multinomial_Bayes_Model.sav')


# In[57]:


Y_pred = classifier_1.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((Y_pred.reshape(len(Y_pred) , 1) , Y_test.reshape(len(Y_test) , 1)) , 1))


# In[58]:


Y_pred = classifier_2.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((Y_pred.reshape(len(Y_pred) , 1) , Y_test.reshape(len(Y_test) , 1)) , 1))


# In[59]:


accuracy_score = classifier_1.predict_proba(X_test)
accuracy_score = accuracy_score.max() * 100
accuracy_score


# In[60]:


accuracy_score = classifier_2.predict_proba(X_test)
accuracy_score = accuracy_score.max() * 100
accuracy_score


# In[61]:


symptoms = dataset_training.columns[ : -2]
print(len(symptoms))


# In[62]:


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


# In[63]:


# NAME OF SYMPTOMS
for i in range(0 , len(input_symtoms)):
    if input_symtoms[i] == 1:
        print(symptoms[i])


# In[64]:


my_symptoms_pred_1 = classifier_1.predict([input_symtoms])
my_symptoms_pred_acc_score_1 = classifier_1.predict_proba([input_symtoms])


# In[65]:


my_symptoms_pred_2 = classifier_2.predict([input_symtoms])
my_symptoms_pred_acc_score_2 = classifier_2.predict_proba([input_symtoms])


# In[66]:


for num , disease in zip(Y_train_model , Y_train):
    if int(num) == int(my_symptoms_pred_1[0]):
        print(disease)
        break
else:
    print('No Disease')
print(my_symptoms_pred_acc_score_1.max()*100)


# In[67]:


for num , disease in zip(Y_train_model , Y_train):
    if int(num) == int(my_symptoms_pred_2[0]):
        print(disease)
        break
else:
    print('No Disease')
print(my_symptoms_pred_acc_score_2.max()*100)


# In[68]:


# joblib.dump(classifier_1 , 'Gaussian_Bayes_Model.sav')
# joblib.dump(classifier_2 , 'Multinomial_Bayes_Model.sav')

