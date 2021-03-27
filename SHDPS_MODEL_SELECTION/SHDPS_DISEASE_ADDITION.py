#!/usr/bin/env python
# coding: utf-8

# # PROGRAM_TO_ADD_DISEASES_TO_DATASET 

# # PROGRAM SHOULD BE ABLE TO PERFORM - 
# ##      -> See Symtomps arranged ALPHABETICALLY
# ##      -> Search For Symptoms
# ##      -> Add Symptoms
# ##      -> Add Disease
# ##      -> Update Disease_Symtoms or Disease_Name or Symptom_Name
# ##      -> Delete DISEASE or SYMPTOM
# ##      -> Update Dataset On Exit

import numpy as np
import pandas as pd
import os , shutil
import re
from prettytable import PrettyTable
from sklearn.naive_bayes import MultinomialNB
from Decision_Tree.decision_tree import train_decision_tree_model
from K_Nearest_Neighbors.knn import train_KNN_model
from Kernel_SVM.kernel_svm import train_SVC_model
from Logistic_Regression.logistic_regression import train_Logistic_regression_model
from Naive_Bayes.naive_bayes import train_naive_bayes_model
from Random_Forest.random_forest import train_random_forest_model


def clear():
    os.system('cls')


def save_on_exit(num , dataset_copy):
    if not os.path.exists(os.getcwd() + '\Datasets\SHDPS_ARCHIVE'):
        os.mkdir(os.getcwd() + '\Datasets\SHDPS_ARCHIVE')
    if not dataset.equals(dataset_copy):
        shutil.move(ORIGINAL , DESTINATION)
        dataset_copy.to_csv(f'Datasets/SHDPS_Training_{num+1}.csv')
        print("---------DATASET SAVED SUCCESSFULLY-----------\nThank's For Using...")
        exit()
    else:
        # clear()
        print("---------DATASET Wasn't ALTERED-----------\nThank's For Using...")
        exit()


def train_all_ML_models(dataset_copy):
    clear()
    print('Processing.........\nPlease Wait.')
    train_decision_tree_model(dataset_copy)
    train_KNN_model(dataset_copy)
    train_SVC_model(dataset_copy)
    train_Logistic_regression_model(dataset_copy)
    train_naive_bayes_model(dataset_copy)
    train_random_forest_model(dataset_copy)
    clear()
    print('-----------ALL MODELS TRAINED SUCCESSFULLY---------------')


# ### TASK -> See Symptoms Arranged ALPHABETICALLY

def see_symptoms():
    clear()
    symptoms = list(dataset_copy.columns[1:])
    first_alpha = set([symptom[0] for symptom in symptoms])
    ask_first_alpha = input('Enter the First Alphabet of symptom which you want to look\n :: ').lower()
    while ask_first_alpha not in first_alpha:
        print('----------No Symptom Found--------------')
        ask_first_alpha = input('Enter the First Alphabet of Symptom which you want to look\n :: ').lower()
    myTable = PrettyTable([ 'ID' , f'SYMPTOMS Starting with "{ask_first_alpha.upper()}"'])
    for index , symptom in enumerate(symptoms):
        if symptom.startswith(ask_first_alpha):
            myTable.add_row([index+1 , symptom])
    clear()
    print(myTable)


# ### TASK -> SEARCH for SYMPTOMS 

def search_symptoms():
    clear()
    asked_symptom = input('Enter the Symtom which you want to Search\n :: ').lower()
    retrived_matches = list(set(list(dataset.filter(like=asked_symptom , axis=1).columns) + list(dataset.filter(regex=f'^{asked_symptom}' , axis = 1).columns) + list(dataset.filter(regex=f'^{asked_symptom[0:2]}' , axis = 1).columns)))
    if len(retrived_matches) != 0:
        myTable = PrettyTable([ 'ID' , f'Symptoms matching with "{asked_symptom.upper()}"'])
        for index , row in enumerate(retrived_matches):
            myTable.add_row([index+1 , row])
        print(myTable)
    else:
        print('\t---------No Matches Found------------')


# ### TASK -> ADD SYMPTOM TO DATASET

def add_symptom():
    clear()
    symptom_to_add = input("Enter Symptom to ADD (PLEASE VERIFY IF IT'S PRESENT)\n :: ").lower()
    symptom_to_add = '_'.join(symptom_to_add.split())
    if symptom_to_add in dataset_copy.columns[1:]:
        print('----------Symptom Already Present----------------')
    else:
        dataset_copy[symptom_to_add] = 0
        print('-------------Symptom Successfully ADDED------------')


# ### TASK -> ADD DISEASE 

# dataset_copy.loc[dataset_copy.shape[0]] = ['Cancer'] + [0] * len(dataset_copy.columns[1:])
# dataset_copy.drop(index=4920 , inplace=True)

def first_character_approach(disease):
    clear()
    user_decision = 'Y'
    symptoms_to_add = []
    while True:
        clear()
        user_decision = input(f'Do you want to ADD MORE Symptoms for {disease.upper()} ( Y- YES , N - NO) (MINIMUM - 5)\n :: ').upper()
        if user_decision != 'Y':
            break
        symptoms = list(dataset_copy.columns[1:])
        first_alpha = set([symptom[0] for symptom in symptoms])
        ask_first_alpha = input('Enter the First Alphabet of symptom which you want to look\n :: ').lower()
        while ask_first_alpha not in first_alpha:
            print('----------No Symptom Found--------------')
            ask_first_alpha = input('Enter the First Alphabet of Symptom which you want to look\n :: ').lower()
        myTable = PrettyTable([ 'ID' , f'SYMPTOMS Starting with "{ask_first_alpha.upper()}"'])
        for index , symptom in enumerate(symptoms):
            if symptom.startswith(ask_first_alpha):
                myTable.add_row([index+1 , symptom])
        clear()
        myTable.sortby = f'SYMPTOMS Starting with "{ask_first_alpha.upper()}"'
        myTable.align[f'SYMPTOMS Starting with "{ask_first_alpha.upper()}"'] = 'r'
        print(myTable)
        symptoms_to_add_str = input('Enter the ID Corresponding to the SYMPTOMS which you want to ADD("SPACE SEPARATED") or PRESS ENTER NOT\n :: ') or None
        if symptoms_to_add_str != None:
            symptoms_to_add_str = list(map(int , symptoms_to_add_str.split()))
            id_check = True
            for symptom_id in symptoms_to_add_str:
                if symptom_id not in range(1 , len(dataset_copy.columns[1:])+1):
                    id_check = False
            if not id_check:
                clear()
                print("-----------One or More ID's was OUT OF RANGE----------------- !\n TRY AGAIN!!")
            else:
                symptoms_to_add = symptoms_to_add + symptoms_to_add_str
    symptoms_names = [0]*len(dataset_copy.columns[1:])
    for i in range(1 , len(dataset_copy.columns[1:])+1):
        if i in symptoms_to_add:
            symptoms_names[i-1] = 1
    return symptoms_names


def full_symptom_table_approach():
    clear()
    all_symtoms = enumerate(dataset_copy.columns[1:])
    myTable = PrettyTable(['ID' , 'ALL SYMPTOMS'])
    for row in all_symtoms:
        myTable.add_row([row[0]+1 , row[1]])
    myTable.sortby = 'ALL SYMPTOMS'
    myTable.align['ALL SYMPTOMS'] = 'r'
    print(myTable)
    symptoms_to_add = input('Enter the ID Corresponding to the SYMPTOMS which you want to ADD("SPACE SEPARATED")\n :: ')
    symptoms_to_add = list(map(int , symptoms_to_add.split()))
    id_check = True
    for symptom_id in symptoms_to_add:
        if symptom_id not in range(1 , len(dataset_copy.columns[1:])+1):
            id_check = False
    if not id_check:
        clear()
        print("-----------One or More ID's was OUT OF RANGE----------------- !\n TRY AGAIN!!")
        full_symptom_table_approach()
    else:
        symptoms_names = [0]*len(dataset_copy.columns[1:])
        for i in range(1 , len(dataset_copy.columns[1:])+1):
            if i in symptoms_to_add:
                symptoms_names[i-1] = 1
#                 symptoms_names[i-1] = dataset_copy.columns[1:][i-1]
        return symptoms_names


def add_disease():
    clear()
    disease_to_add = input('Enter the Name of the Disease\n :: ').title()
    choice_1 = input(f'Do you want to ADD symptoms for {disease_to_add} (Y - YES , X - DISCARD)\n :: ').upper()
    choice_2 = input('-----------AVAILABLE METHODS-------------\n\n 1. First Character Approach\n 2. FULL SYMPTOM TABLE APPROACH\n :: ')
    while (choice_1 not in ['Y' , 'X']) or not choice_2.isnumeric():
        clear()
        print('----------Either of the input was INVALID !------------')
        choice_1 = input(f'Do you want to ADD symptoms for {disease_to_add} (Y - YES , X - DISCARD) :: ').upper() or 'Y'
        choice_2 = input('-----------AVAILABLE METHODS-------------\n\n 1. First Character Approach\n 2. FULL SYMPTOM TABLE APPROACH\n') or '1'
    if choice_1 == 'Y' and choice_2 == '1':
        symptoms_to_add = first_character_approach(disease_to_add)
        if symptoms_to_add.count(1) >= 4:
            dataset_copy.loc[dataset_copy.shape[0]] = [disease_to_add] + symptoms_to_add
            clear()
            print('-------------DISEASE ADDED SUCCESSFULLY-------------')
        else:
            print(f'---------Minimum Five Symptoms essential for Registering {disease_to_add}----------')
    elif choice_1 == 'Y' and choice_2 == '2':
        symptoms_to_add = full_symptom_table_approach()
        if symptoms_to_add.count(1) >= 4:
            dataset_copy.loc[dataset_copy.shape[0]] = [disease_to_add] + symptoms_to_add
            clear()
            print('-------------DISEASE ADDED SUCCESSFULLY-------------')
        else:
            print(f'---------Minimum Five Symptoms essential for Registering {disease_to_add}----------')
    else:
        clear()
        print('-------Disease was DISCARDED---------')

        

if __name__ == "__main__":
    dirs = os.listdir('Datasets/')
    dirs = dirs[1]
    num = int(re.findall('[0-9]+' , dirs)[0])
    ORIGINAL = os.getcwd() + f'\Datasets\SHDPS_Training_{num}.csv'
    DESTINATION = os.getcwd() + f'\Datasets\SHDPS_ARCHIVE/SHDPS_Training_{num}(ARCHIVED).csv'
    # dataset = pd.read_csv(ORIGINAL , index_col='prognosis')
    headers = [*pd.read_csv(ORIGINAL, nrows=1)]
    dataset = pd.read_csv(ORIGINAL, usecols=[c for c in headers if c != 'Unnamed: 0'])
    dataset_copy = dataset.copy(deep=True)

    while True:
        user_input = input('\t------------PROGRAM TO ADD DISEASES TO SHDPS DATASET-----------------\n  1. SEE AVAILABLE SYMPTOMS\n  2. SEARCH FROM AVAILABLE SYMPTOMS\n  3. ADD SYMPTOM\n  4. ADD DISEASE\n  5. RE-TRAIN ALL ML MODELS\n  6. SAVE & EXIT\n  :: ')
        if user_input == '1':
            see_symptoms()
        elif user_input == '2':
            search_symptoms()
        elif user_input == '3':
            add_symptom()
        elif user_input == '4':
            add_disease()
        elif user_input == '5':
            train_all_ML_models(dataset_copy=dataset_copy)
        elif user_input == '6':
            save_on_exit(num=num , dataset_copy=dataset_copy)
        else:
            clear()
            print('--------INVALID INPUT-----------')